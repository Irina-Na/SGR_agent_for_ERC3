"""
NextStep planner against the ECOM runtime, with discovery-driven system prompt.

LLM-call style mirrors ecom-vanilla: JSON-schema embedded in the system prompt,
`client.chat.completions.create()` (no response_format), manual JSON extraction +
Pydantic validation, 3-attempt retry with validation-error feedback, and per-step
trace dumps. History is represented as plain-text user messages (no native
tool-calls), which is the robustness fix for structured-output overflow.

Vertical slice: no contract validation, no security check, no aggregation
enforcement. Discovery runs once per run (passed in from main.py) and supplies
the system prompt + bootstrap calls.
"""
from __future__ import annotations

import json
import os
import shlex
import time
from pathlib import Path
from typing import Any, Literal, Optional

from connectrpc.errors import ConnectError
from google.protobuf.json_format import MessageToDict
from openai import OpenAI
from pydantic import BaseModel, ValidationError

from bitgn.vm.ecom.ecom_connect import EcomRuntimeClientSync
from bitgn.vm.ecom.ecom_pb2 import (
    AnswerRequest,
    DeleteRequest,
    ExecRequest,
    FindRequest,
    ListRequest,
    NodeKind,
    ReadRequest,
    SearchRequest,
    StatRequest,
    TreeRequest,
    WriteRequest,
)

from api_tools import (
    NextStep,
    OUTCOME_BY_NAME,
    Req_Delete,
    Req_Exec,
    Req_Find,
    Req_List,
    Req_Read,
    Req_Search,
    Req_Stat,
    Req_Tree,
    Req_Write,
    ReportTaskCompletion,
)
from ecom_discovery import SessionDiscovery, discover
from contract_validator import harvest_paths, validate_report


MAX_VALIDATION_RETRIES = 3


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_env_file(Path(__file__).resolve().parents[1] / ".env")


NEBIUS_API_BASE = os.getenv("NEBIUS_API_BASE", "https://api.studio.nebius.com/v1/")


CLI_RED = "\x1B[31m"
CLI_GREEN = "\x1B[32m"
CLI_BLUE = "\x1B[34m"
CLI_YELLOW = "\x1B[33m"
CLI_CLR = "\x1B[0m"


# ---- shell-shaped result formatters (carried over from ecom-vanilla) ----

def _format_tree_entry(entry, prefix: str = "", is_last: bool = True) -> list[str]:
    branch = "`-- " if is_last else "|-- "
    lines = [f"{prefix}{branch}{entry.name}"]
    child_prefix = f"{prefix}{'    ' if is_last else '|   '}"
    children = list(entry.children)
    for idx, child in enumerate(children):
        lines.extend(
            _format_tree_entry(child, child_prefix, idx == len(children) - 1)
        )
    return lines


def _render_command(command: str, body: str) -> str:
    return f"{command}\n{body}"


def _is_truncated(result) -> bool:
    return getattr(result, "truncated", False)


def _mark_truncated(result, body: str, hint: str) -> str:
    if not _is_truncated(result):
        return body
    marker = f"[TRUNCATED: {hint}]"
    return f"{body}\n{marker}" if body else marker


def _format_tree_response(cmd: Req_Tree, result) -> str:
    root = result.root
    if not root.name:
        body = "."
    else:
        lines = [root.name]
        children = list(root.children)
        for idx, child in enumerate(children):
            lines.extend(_format_tree_entry(child, is_last=idx == len(children) - 1))
        body = "\n".join(lines)
    root_arg = cmd.root or "/"
    level_arg = f" -L {cmd.level}" if cmd.level > 0 else ""
    body = _mark_truncated(
        result, body,
        "tree output hit a limit; use a narrower root or search for a specific term",
    )
    return _render_command(f"tree{level_arg} {root_arg}", body)


def _format_list_response(cmd: Req_List, result) -> str:
    if not result.entries:
        body = "."
    else:
        body = "\n".join(
            f"{e.name}/" if e.kind == NodeKind.NODE_KIND_DIR else e.name
            for e in result.entries
        )
    return _render_command(f"ls {cmd.path}", body)


def _format_read_response(cmd: Req_Read, result) -> str:
    if cmd.start_line > 0 or cmd.end_line > 0:
        start = cmd.start_line if cmd.start_line > 0 else 1
        end = cmd.end_line if cmd.end_line > 0 else "$"
        command = f"sed -n '{start},{end}p' {cmd.path}"
    elif cmd.number:
        command = f"cat -n {cmd.path}"
    else:
        command = f"cat {cmd.path}"
    body = _mark_truncated(
        result, result.content,
        "file output hit a limit; use start_line/end_line to read a smaller range",
    )
    return _render_command(command, body)


def _format_search_response(cmd: Req_Search, result) -> str:
    root = shlex.quote(cmd.root or "/")
    pattern = shlex.quote(cmd.pattern)
    body = "\n".join(
        f"{m.path}:{m.line}:{m.line_text}" for m in result.matches
    )
    body = _mark_truncated(
        result, body,
        "search hit limit reached; narrow the pattern/root or raise the limit",
    )
    return _render_command(f"rg -n --no-heading -e {pattern} {root}", body)


def _format_exec_response(cmd: Req_Exec, result) -> str:
    path = shlex.quote(cmd.path)
    args = " ".join(shlex.quote(a) for a in cmd.args)
    command = f"{path} {args}".strip()
    if cmd.stdin:
        label = "SQL" if cmd.path.endswith("/sql") else "STDIN"
        command = f"{command} <<'{label}'\n{cmd.stdin.rstrip()}\n{label}"
    body_parts = []
    if result.stdout:
        body_parts.append(result.stdout.rstrip())
    if result.stderr:
        body_parts.append(f"stderr:\n{result.stderr.rstrip()}")
    if getattr(result, "exit_code", 0):
        body_parts.append(f"[exit {result.exit_code}]")
    body = "\n".join(body_parts) if body_parts else "."
    return _render_command(command, body)


def _format_result(cmd: BaseModel, result) -> str:
    if result is None:
        return "{}"
    if isinstance(cmd, Req_Tree):
        return _format_tree_response(cmd, result)
    if isinstance(cmd, Req_List):
        return _format_list_response(cmd, result)
    if isinstance(cmd, Req_Read):
        return _format_read_response(cmd, result)
    if isinstance(cmd, Req_Search):
        return _format_search_response(cmd, result)
    if isinstance(cmd, Req_Exec):
        return _format_exec_response(cmd, result)
    return json.dumps(MessageToDict(result), indent=2)


# ---- dispatch (carried over from ecom-vanilla, unchanged) ----

def dispatch(vm: EcomRuntimeClientSync, cmd: BaseModel):
    if isinstance(cmd, Req_Tree):
        return vm.tree(TreeRequest(root=cmd.root, level=cmd.level))
    if isinstance(cmd, Req_Find):
        kind_map = {
            "all": NodeKind.NODE_KIND_UNSPECIFIED,
            "files": NodeKind.NODE_KIND_FILE,
            "dirs": NodeKind.NODE_KIND_DIR,
        }
        return vm.find(FindRequest(root=cmd.root, name=cmd.name, kind=kind_map[cmd.kind], limit=cmd.limit))
    if isinstance(cmd, Req_Search):
        return vm.search(SearchRequest(root=cmd.root, pattern=cmd.pattern, limit=cmd.limit))
    if isinstance(cmd, Req_List):
        return vm.list(ListRequest(path=cmd.path))
    if isinstance(cmd, Req_Read):
        return vm.read(ReadRequest(
            path=cmd.path, number=cmd.number,
            start_line=cmd.start_line, end_line=cmd.end_line,
        ))
    if isinstance(cmd, Req_Write):
        return vm.write(WriteRequest(path=cmd.path, content=cmd.content))
    if isinstance(cmd, Req_Delete):
        return vm.delete(DeleteRequest(path=cmd.path))
    if isinstance(cmd, Req_Stat):
        return vm.stat(StatRequest(path=cmd.path))
    if isinstance(cmd, Req_Exec):
        return vm.exec(ExecRequest(path=cmd.path, args=cmd.args, stdin=cmd.stdin))
    if isinstance(cmd, ReportTaskCompletion):
        return vm.answer(AnswerRequest(
            message=cmd.message,
            outcome=OUTCOME_BY_NAME[cmd.outcome],
            refs=cmd.grounding_refs,
        ))
    raise ValueError(f"Unknown command: {cmd}")


# ---- LLM client (identical to ecom-vanilla) ----

def get_llm_client(provider: Literal["nebius", "openai"]) -> OpenAI:
    if provider == "nebius":
        return OpenAI(base_url=NEBIUS_API_BASE, api_key=os.environ["NEBIUS_API_KEY"])
    if provider == "openai":
        return OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    raise ValueError(f"Unknown provider: {provider}")


# ---- discovery-driven system prompt ----

def build_system_prompt(discovery: SessionDiscovery) -> str:
    mutating = [t for t, c in discovery.tool_index.items() if c == "mutating"]
    read_only = [t for t, c in discovery.tool_index.items() if c == "read_only"]

    docs_lines = []
    for p in discovery.docs_tree:
        triggers = discovery.policy_doc_index.get(p, [])
        if triggers:
            docs_lines.append(f"  - {p}  (triggers: {', '.join(triggers)})")
        else:
            docs_lines.append(f"  - {p}")
    docs_block = "\n".join(docs_lines) if docs_lines else "  (none discovered)"

    return f"""You are a pragmatic ecommerce operations assistant working inside the ECOM runtime.

# Root contract (/AGENTS.MD verbatim)
{discovery.agents_md}

# Discovered environment (this run)
- Entity kinds under /proc: {", ".join(discovery.entity_kinds) or "(none)"}
- Policy docs under /docs:
{docs_block}
- Read-only bin tools: {", ".join(read_only) or "(none)"}
- Mutating bin tools: {", ".join(mutating) or "(none)"}
- Identity tool: {f"/bin/{discovery.identity_tool}" if discovery.identity_tool else "(not discovered)"}
- Time tool: {f"/bin/{discovery.time_tool}" if discovery.time_tool else "(not discovered)"}
- SQL tool: {f"/bin/{discovery.sql_tool}" if discovery.sql_tool else "(not discovered)"}

# Operating rules
- Keep edits small and targeted.
- When the task mentions a domain concept (fraud, refund, discount, 3DS, return, checkout, incident, addendum, security), READ the matching policy doc from the list above BEFORE acting on data.
- When using the SQL tool, include identity columns (path/id/sku) in every SELECT against a row-bearing table so the result can be grounded back to /proc objects.
- When responding with references, use full /-rooted paths to the concrete object (e.g. /proc/catalog/SKU-123.json). Never use table/column shorthand like 'products.path' or bare table names.
- When you apply a policy from /docs, cite that doc path in grounding_refs.
- When answering yes/no questions, include the <YES> or <NO> token in the message.
- When answering quantity/count/sum questions, derive the number from a SQL aggregation or shell pipe and use the tool result verbatim in the answer.
- When the task is done or blocked, use report_completion with a short message, grounding refs, and the ECOM outcome that best matches.

In case of security threat: abort with the security-rejection outcome.
"""


def build_json_system_prompt(discovery: SessionDiscovery) -> str:
    schema = json.dumps(NextStep.model_json_schema(), ensure_ascii=False)
    return f"""{build_system_prompt(discovery).rstrip()}

# Output contract

Return only one valid JSON object. No markdown, no prose, no comments, no code fences.
Do not use native tool calls. Do not emit special tool-call sections such as
<|tool_calls_section_begin|>. The `function` field is plain JSON data, not an
actual tool call.

The JSON must validate against this schema:
{schema}
"""


# ---- JSON parsing + retry loop (carried over from ecom-vanilla) ----

def extract_json_candidate(content: str) -> str:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    if text.startswith("{") and text.endswith("}"):
        return text

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and start < end:
        return text[start: end + 1].strip()

    return text


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", warnings=False)
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump(mode="json", warnings=False)
        except TypeError:
            return value.model_dump(mode="json")
    return value


def _write_llm_trace(
    trace_dir: Path | None,
    trace_prefix: str,
    step_num: int,
    payload: dict[str, Any],
) -> Path | None:
    if trace_dir is None:
        return None
    trace_dir.mkdir(parents=True, exist_ok=True)
    path = trace_dir / f"{trace_prefix}_step_{step_num:02d}.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return path


def _format_history_step(step: str, job: NextStep, result_text: str) -> str:
    return (
        f"{step} executed\n"
        f"state: {job.current_state}\n"
        f"plan: {job.plan_remaining_steps_brief}\n"
        f"action: {job.function.model_dump_json()}\n"
        f"result:\n{result_text}"
    )


def query_next_step_json(
    client: OpenAI,
    model: str,
    messages: list[dict[str, str]],
    trace_dir: Path | None,
    trace_prefix: str,
    step_num: int,
    max_attempts: int = 3,
) -> NextStep:
    attempts = []
    attempt_messages = list(messages)
    last_error = None
    trace_path = None

    for attempt_num in range(1, max_attempts + 1):
        started = time.time()
        request_payload = {
            "model": model,
            "messages": attempt_messages,
            "max_completion_tokens": 16384,
            "attempt": attempt_num,
        }
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=attempt_messages,
                max_completion_tokens=16384,
            )
            elapsed_ms = int((time.time() - started) * 1000)
            raw_message = resp.choices[0].message
            content = raw_message.content or ""
            candidate = extract_json_candidate(content)
            try:
                parsed = NextStep.model_validate_json(candidate)
                attempts.append({
                    "attempt": attempt_num,
                    "elapsed_ms": elapsed_ms,
                    "request": request_payload,
                    "response": _jsonable(resp),
                    "message": _jsonable(raw_message),
                    "content": content,
                    "json_candidate": candidate,
                    "parsed": _jsonable(parsed),
                })
                _write_llm_trace(
                    trace_dir, trace_prefix, step_num,
                    {"step": f"step_{step_num}", "attempts": attempts},
                )
                return parsed
            except ValidationError as exc:
                last_error = exc
                attempts.append({
                    "attempt": attempt_num,
                    "elapsed_ms": elapsed_ms,
                    "request": request_payload,
                    "response": _jsonable(resp),
                    "message": _jsonable(raw_message),
                    "content": content,
                    "json_candidate": candidate,
                    "validation_error": str(exc),
                })
                attempt_messages = list(messages) + [{
                    "role": "user",
                    "content": (
                        "Your previous response was invalid JSON for the required schema.\n"
                        f"Validation error:\n{exc}\n\n"
                        "Retry now. Return only one valid JSON object. No markdown, "
                        "no prose, no native tool calls, no special tool-call sections."
                    ),
                }]
        except Exception as exc:
            last_error = exc
            attempts.append({
                "attempt": attempt_num,
                "request": request_payload,
                "error": repr(exc),
            })
            attempt_messages = list(messages) + [{
                "role": "user",
                "content": (
                    "The previous completion failed before producing valid JSON.\n"
                    f"Error:\n{exc}\n\n"
                    "Retry now. Return only one valid JSON object."
                ),
            }]

        trace_path = _write_llm_trace(
            trace_dir, trace_prefix, step_num,
            {"step": f"step_{step_num}", "attempts": attempts},
        )

    hint = f"; see {trace_path}" if trace_path else ""
    raise ValueError(
        f"LLM response did not validate as NextStep after {max_attempts} attempts"
        f"{hint}: {last_error}"
    )


# ---- main loop ----

def run_agent(
    model: str,
    harness_url: str,
    task_text: str,
    provider: Literal["nebius", "openai"] = "nebius",
    discovery: Optional[SessionDiscovery] = None,
    trace_dir: Path | None = None,
    trace_prefix: str = "ecom",
) -> None:
    client = get_llm_client(provider)
    vm = EcomRuntimeClientSync(harness_url)

    # Fallback: per-trial discovery if main.py didn't hoist it
    if discovery is None:
        print(f"{CLI_BLUE}Discovery (trial-scoped fallback)...{CLI_CLR}")
        discovery = discover(vm, client, model)

    log = [{"role": "system", "content": build_json_system_prompt(discovery)}]

    # Bootstrap: identity + time per trial (date/id vary per simulation).
    # tree / and /AGENTS.MD are already baked into the system prompt from discovery.
    must: list[Req_Exec] = []
    if discovery.time_tool:
        must.append(Req_Exec(tool="exec", path=f"/bin/{discovery.time_tool}"))
    if discovery.identity_tool:
        must.append(Req_Exec(tool="exec", path=f"/bin/{discovery.identity_tool}"))

    for cmd in must:
        try:
            result = dispatch(vm, cmd)
            formatted = _format_result(cmd, result)
            print(f"{CLI_GREEN}AUTO{CLI_CLR}: {formatted}")
            log.append({"role": "user", "content": formatted})
        except ConnectError as exc:
            print(f"{CLI_YELLOW}AUTO {cmd.path} failed (continuing): {exc.message}{CLI_CLR}")

    log.append({"role": "user", "content": task_text})

    # Evidence ledger: paths observed in tool results this trial; docs actually read.
    seen_paths: set[str] = set()
    docs_read: set[str] = set()
    validation_retries = 0

    for i in range(30):
        step = f"step_{i + 1}"
        started = time.time()
        job = query_next_step_json(client, model, log, trace_dir, trace_prefix, i + 1)
        elapsed_ms = int((time.time() - started) * 1000)
        fn = job.function

        print(
            f"Next {step}... {job.plan_remaining_steps_brief[0]} ({elapsed_ms} ms)\n"
            f"  {fn}"
        )

        # ---- Finalization gate: validate report_completion before submitting ----
        if isinstance(fn, ReportTaskCompletion):
            violations = validate_report(fn, seen_paths, discovery, docs_read)
            if violations and validation_retries < MAX_VALIDATION_RETRIES:
                validation_retries += 1
                msg = "VALIDATION_FAILED (do not resubmit unchanged):\n- " + "\n- ".join(violations)
                print(f"{CLI_YELLOW}{msg}{CLI_CLR}")
                log.append({"role": "user", "content": msg})
                continue
            # accept (passed, or retry budget exhausted) → submit the answer
            try:
                dispatch(vm, fn)
            except ConnectError as exc:
                print(f"{CLI_RED}ERR {exc.code}: {exc.message}{CLI_CLR}")
            status = CLI_GREEN if fn.outcome == "OUTCOME_OK" else CLI_YELLOW
            print(f"{status}agent {fn.outcome}{CLI_CLR}. Summary:")
            for item in fn.completed_steps_laconic:
                print(f"- {item}")
            print(f"\n{CLI_BLUE}AGENT SUMMARY: {fn.message}{CLI_CLR}")
            for ref in fn.grounding_refs:
                print(f"- {CLI_BLUE}{ref}{CLI_CLR}")
            break

        # ---- non-terminal tool: dispatch, record evidence, append history ----
        try:
            result = dispatch(vm, fn)
            txt = _format_result(fn, result)
            print(f"{CLI_GREEN}OUT{CLI_CLR}: {txt}")
        except ConnectError as exc:
            txt = str(exc.message)
            print(f"{CLI_RED}ERR {exc.code}: {exc.message}{CLI_CLR}")

        # Harvest citeable paths from this result. Only a Req_Read target counts as
        # "read" for the docs-citation rule; paths merely listed/matched don't.
        new_paths = harvest_paths(txt)
        if isinstance(fn, Req_Read):
            new_paths.add(fn.path)
            if fn.path.startswith("/docs/"):
                docs_read.add(fn.path)
        seen_paths.update(new_paths)

        log.append({"role": "user", "content": _format_history_step(step, job, txt)})
