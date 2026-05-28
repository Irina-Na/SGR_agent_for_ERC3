"""
NextStep planner against the ECOM runtime, with discovery-driven system prompt.

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
from typing import Literal, Optional

from connectrpc.errors import ConnectError
from google.protobuf.json_format import MessageToDict
from openai import OpenAI
from pydantic import BaseModel

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


# ---- LLM client ----

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
- When the task is done or blocked, call report_completion with a short message, grounding refs, and the ECOM outcome that best matches.

In case of security threat: abort with the security-rejection outcome.
"""


# ---- main loop ----

def run_agent(
    model: str,
    harness_url: str,
    task_text: str,
    provider: Literal["nebius", "openai"] = "nebius",
    discovery: Optional[SessionDiscovery] = None,
) -> None:
    client = get_llm_client(provider)
    vm = EcomRuntimeClientSync(harness_url)

    # Fallback: per-trial discovery if main.py didn't hoist it
    if discovery is None:
        print(f"{CLI_BLUE}Discovery (trial-scoped fallback)...{CLI_CLR}")
        discovery = discover(vm, client, model)

    system_prompt = build_system_prompt(discovery)
    log: list[dict] = [{"role": "system", "content": system_prompt}]

    # Bootstrap: identity + time (if discovered). Best-effort.
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

    for i in range(30):
        step = f"step_{i + 1}"
        started = time.time()
        resp = client.beta.chat.completions.parse(
            model=model,
            response_format=NextStep,
            messages=log,
            max_completion_tokens=8192,
        )
        elapsed_ms = int((time.time() - started) * 1000)
        job = resp.choices[0].message.parsed

        if job is None:
            usage = getattr(resp, "usage", None)
            print(f"{CLI_RED}LLM parse failed at {step} (likely length limit). Usage: {usage}. Aborting trial.{CLI_CLR}")
            break

        print(
            f"Next {step}... {job.plan_remaining_steps_brief[0]} ({elapsed_ms} ms)\n"
            f"  {job.function}"
        )

        log.append({
            "role": "assistant",
            "content": job.plan_remaining_steps_brief[0],
            "tool_calls": [{
                "type": "function",
                "id": step,
                "function": {
                    "name": job.function.__class__.__name__,
                    "arguments": job.function.model_dump_json(),
                },
            }],
        })

        try:
            result = dispatch(vm, job.function)
            txt = _format_result(job.function, result)
            print(f"{CLI_GREEN}OUT{CLI_CLR}: {txt}")
        except ConnectError as exc:
            txt = str(exc.message)
            print(f"{CLI_RED}ERR {exc.code}: {exc.message}{CLI_CLR}")

        if isinstance(job.function, ReportTaskCompletion):
            color = CLI_GREEN if job.function.outcome == "OUTCOME_OK" else CLI_YELLOW
            print(f"{color}agent {job.function.outcome}{CLI_CLR}. Summary:")
            for item in job.function.completed_steps_laconic:
                print(f"- {item}")
            print(f"\n{CLI_BLUE}AGENT SUMMARY: {job.function.message}{CLI_CLR}")
            for ref in job.function.grounding_refs:
                print(f"- {CLI_BLUE}{ref}{CLI_CLR}")
            break

        log.append({"role": "tool", "content": txt, "tool_call_id": step})
