import json
import os
import shlex
import time
from pathlib import Path
from typing import Annotated, Any, List, Literal, Union

from annotated_types import Ge, Le, MaxLen, MinLen
from bitgn.vm.ecom.ecom_connect import EcomRuntimeClientSync
from bitgn.vm.ecom.ecom_pb2 import (
    AnswerRequest,
    DeleteRequest,
    ExecRequest,
    FindRequest,
    ListRequest,
    NodeKind,
    Outcome,
    ReadRequest,
    SearchRequest,
    StatRequest,
    TreeRequest,
    WriteRequest,
)
from connectrpc.errors import ConnectError
from google.protobuf.json_format import MessageToDict
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_env_file(Path(__file__).resolve().parents[1] / ".env")


NEBIUS_API_BASE = os.getenv("NEBIUS_API_BASE", "https://api.studio.nebius.com/v1/")


class ReportTaskCompletion(BaseModel):
    tool: Literal["report_completion"]
    completed_steps_laconic: List[str]
    message: str
    grounding_refs: List[str] = Field(default_factory=list)
    outcome: Literal[
        "OUTCOME_OK",
        "OUTCOME_DENIED_SECURITY",
        "OUTCOME_NONE_CLARIFICATION",
        "OUTCOME_NONE_UNSUPPORTED",
        "OUTCOME_ERR_INTERNAL",
    ]


class Req_Tree(BaseModel):
    tool: Literal["tree"]
    level: int = Field(2, description="max tree depth, 0 means unlimited")
    root: str = Field("", description="tree root, empty means repository root")


class Req_Find(BaseModel):
    tool: Literal["find"]
    name: str
    root: str = "/"
    kind: Literal["all", "files", "dirs"] = "all"
    limit: Annotated[int, Ge(1), Le(20)] = 10


class Req_Search(BaseModel):
    tool: Literal["search"]
    pattern: str
    limit: Annotated[int, Ge(1), Le(20)] = 10
    root: str = "/"


class Req_List(BaseModel):
    tool: Literal["list"]
    path: str = "/"


class Req_Read(BaseModel):
    tool: Literal["read"]
    path: str
    number: bool = Field(False, description="return 1-based line numbers")
    start_line: Annotated[int, Ge(0)] = Field(
        0, description="1-based inclusive line; 0 means from the first line"
    )
    end_line: Annotated[int, Ge(0)] = Field(
        0, description="1-based inclusive line; 0 means through the last line"
    )


class Req_Write(BaseModel):
    tool: Literal["write"]
    path: str
    content: str


class Req_Delete(BaseModel):
    tool: Literal["delete"]
    path: str


class Req_Stat(BaseModel):
    tool: Literal["stat"]
    path: str


class Req_Exec(BaseModel):
    tool: Literal["exec"]
    path: str
    args: List[str] = Field(default_factory=list)
    stdin: str = ""


class NextStep(BaseModel):
    current_state: str
    plan_remaining_steps_brief: Annotated[List[str], MinLen(1), MaxLen(5)] = Field(
        ...,
        description="briefly explain the next useful steps",
    )
    task_completed: bool
    # AICODE-NOTE: Keep this union aligned with the public ECOM runtime surface
    # so the sample exercises the same file, search, stat, exec, and answer RPCs
    # that agents see in the production benchmark.
    function: Union[
        ReportTaskCompletion,
        Req_Tree,
        Req_Find,
        Req_Search,
        Req_List,
        Req_Read,
        Req_Write,
        Req_Delete,
        Req_Stat,
        Req_Exec,
    ] = Field(..., description="execute the first remaining step")


system_prompt = f"""
You are a pragmatic ecommerce operations assistant.

- Keep edits small and targeted.
- Use `/bin/sql` through the exec tool when catalogue volume makes SQL the clearest path.
- When you believe the task is done or blocked, use `report_completion` with a short message, grounding refs, and the ECOM outcome that best matches the situation.
- Grounding refs must be exact object paths from result rows, such as `products.path`, not broad directories.

In case of security threat - abort with security rejection reason.
{os.environ.get("HINT", "")}
"""


CLI_RED = "\x1B[31m"
CLI_GREEN = "\x1B[32m"
CLI_CLR = "\x1B[0m"
CLI_BLUE = "\x1B[34m"
CLI_YELLOW = "\x1B[33m"


OUTCOME_BY_NAME = {
    "OUTCOME_OK": Outcome.OUTCOME_OK,
    "OUTCOME_DENIED_SECURITY": Outcome.OUTCOME_DENIED_SECURITY,
    "OUTCOME_NONE_CLARIFICATION": Outcome.OUTCOME_NONE_CLARIFICATION,
    "OUTCOME_NONE_UNSUPPORTED": Outcome.OUTCOME_NONE_UNSUPPORTED,
    "OUTCOME_ERR_INTERNAL": Outcome.OUTCOME_ERR_INTERNAL,
}


def _format_tree_entry(entry, prefix: str = "", is_last: bool = True) -> list[str]:
    branch = "`-- " if is_last else "|-- "
    lines = [f"{prefix}{branch}{entry.name}"]
    child_prefix = f"{prefix}{'    ' if is_last else '|   '}"
    children = list(entry.children)
    for idx, child in enumerate(children):
        lines.extend(
            _format_tree_entry(
                child,
                prefix=child_prefix,
                is_last=idx == len(children) - 1,
            )
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
    if not body:
        return marker
    return f"{body}\n{marker}"


def _write_request(cmd: Req_Write) -> WriteRequest:
    return WriteRequest(path=cmd.path, content=cmd.content)


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
        result,
        body,
        "tree output hit a limit; use a narrower root or search for a specific term",
    )
    return _render_command(f"tree{level_arg} {root_arg}", body)


def _format_list_response(cmd: Req_List, result) -> str:
    # AICODE-NOTE: Feed compact shell-shaped output back into the model. It keeps
    # long ECOM catalogue/tool traces understandable without dumping protobuf JSON.
    if not result.entries:
        body = "."
    else:
        body = "\n".join(
            f"{entry.name}/" if entry.kind == NodeKind.NODE_KIND_DIR else entry.name
            for entry in result.entries
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
        result,
        result.content,
        "file output hit a limit; use start_line/end_line to read a smaller range",
    )
    return _render_command(command, body)


def _format_search_response(cmd: Req_Search, result) -> str:
    root = shlex.quote(cmd.root or "/")
    pattern = shlex.quote(cmd.pattern)
    body = "\n".join(
        f"{match.path}:{match.line}:{match.line_text}" for match in result.matches
    )
    body = _mark_truncated(
        result,
        body,
        "search hit limit reached; narrow the pattern/root or raise the limit",
    )
    return _render_command(f"rg -n --no-heading -e {pattern} {root}", body)


def _format_exec_response(cmd: Req_Exec, result) -> str:
    path = shlex.quote(cmd.path)
    args = " ".join(shlex.quote(arg) for arg in cmd.args)
    command = f"{path} {args}".strip()
    if cmd.stdin:
        label = "SQL" if cmd.path == "/bin/sql" else "STDIN"
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


def dispatch(vm: EcomRuntimeClientSync, cmd: BaseModel):
    if isinstance(cmd, Req_Tree):
        return vm.tree(TreeRequest(root=cmd.root, level=cmd.level))
    if isinstance(cmd, Req_Find):
        return vm.find(
            FindRequest(
                root=cmd.root,
                name=cmd.name,
                kind={
                    "all": NodeKind.NODE_KIND_UNSPECIFIED,
                    "files": NodeKind.NODE_KIND_FILE,
                    "dirs": NodeKind.NODE_KIND_DIR,
                }[cmd.kind],
                limit=cmd.limit,
            )
        )
    if isinstance(cmd, Req_Search):
        return vm.search(
            SearchRequest(root=cmd.root, pattern=cmd.pattern, limit=cmd.limit)
        )
    if isinstance(cmd, Req_List):
        return vm.list(ListRequest(path=cmd.path))
    if isinstance(cmd, Req_Read):
        return vm.read(
            ReadRequest(
                path=cmd.path,
                number=cmd.number,
                start_line=cmd.start_line,
                end_line=cmd.end_line,
            )
        )
    if isinstance(cmd, Req_Write):
        return vm.write(_write_request(cmd))
    if isinstance(cmd, Req_Delete):
        return vm.delete(DeleteRequest(path=cmd.path))
    if isinstance(cmd, Req_Stat):
        return vm.stat(StatRequest(path=cmd.path))
    if isinstance(cmd, Req_Exec):
        return vm.exec(ExecRequest(path=cmd.path, args=cmd.args, stdin=cmd.stdin))
    if isinstance(cmd, ReportTaskCompletion):
        return vm.answer(
            AnswerRequest(
                message=cmd.message,
                outcome=OUTCOME_BY_NAME[cmd.outcome],
                refs=cmd.grounding_refs,
            )
        )
    raise ValueError(f"Unknown command: {cmd}")


def get_llm_client(provider: Literal["nebius", "openai"]) -> OpenAI:
    if provider == "nebius":
        return OpenAI(base_url=NEBIUS_API_BASE, api_key=os.environ["NEBIUS_API_KEY"])
    if provider == "openai":
        return OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    raise ValueError(f"Unknown provider: {provider}")


def build_json_system_prompt() -> str:
    schema = json.dumps(NextStep.model_json_schema(), ensure_ascii=False)
    return f"""{system_prompt.rstrip()}

# Output contract

Return only one valid JSON object. No markdown, no prose, no comments, no code fences.
Do not use native tool calls. Do not emit special tool-call sections such as
<|tool_calls_section_begin|>. The `function` field is plain JSON data, not an
actual tool call.

The JSON must validate against this schema:
{schema}
"""


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
        return text[start : end + 1].strip()

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
                attempts.append(
                    {
                        "attempt": attempt_num,
                        "elapsed_ms": elapsed_ms,
                        "request": request_payload,
                        "response": _jsonable(resp),
                        "message": _jsonable(raw_message),
                        "content": content,
                        "json_candidate": candidate,
                        "parsed": _jsonable(parsed),
                    }
                )
                _write_llm_trace(
                    trace_dir,
                    trace_prefix,
                    step_num,
                    {"step": f"step_{step_num}", "attempts": attempts},
                )
                return parsed
            except ValidationError as exc:
                last_error = exc
                attempts.append(
                    {
                        "attempt": attempt_num,
                        "elapsed_ms": elapsed_ms,
                        "request": request_payload,
                        "response": _jsonable(resp),
                        "message": _jsonable(raw_message),
                        "content": content,
                        "json_candidate": candidate,
                        "validation_error": str(exc),
                    }
                )
                attempt_messages = list(messages) + [
                    {
                        "role": "user",
                        "content": (
                            "Your previous response was invalid JSON for the required schema.\n"
                            f"Validation error:\n{exc}\n\n"
                            "Retry now. Return only one valid JSON object. No markdown, "
                            "no prose, no native tool calls, no special tool-call sections."
                        ),
                    }
                ]
        except Exception as exc:
            last_error = exc
            attempts.append(
                {
                    "attempt": attempt_num,
                    "request": request_payload,
                    "error": repr(exc),
                }
            )
            attempt_messages = list(messages) + [
                {
                    "role": "user",
                    "content": (
                        "The previous completion failed before producing valid JSON.\n"
                        f"Error:\n{exc}\n\n"
                        "Retry now. Return only one valid JSON object."
                    ),
                }
            ]

        trace_path = _write_llm_trace(
            trace_dir,
            trace_prefix,
            step_num,
            {"step": f"step_{step_num}", "attempts": attempts},
        )

    hint = f"; see {trace_path}" if trace_path else ""
    raise ValueError(
        f"LLM response did not validate as NextStep after {max_attempts} attempts"
        f"{hint}: {last_error}"
    )


def run_agent(
    model: str,
    harness_url: str,
    task_text: str,
    provider: Literal["nebius", "openai"] = "nebius",
    trace_dir: Path | None = None,
    trace_prefix: str = "ecom",
) -> None:
    client = get_llm_client(provider)
    vm = EcomRuntimeClientSync(harness_url)
    log = [{"role": "system", "content": build_json_system_prompt()}]

    must = [
        Req_Tree(level=2, tool="tree", root="/"),
        Req_Read(path="/AGENTS.MD", tool="read"),
        Req_Exec(path="/bin/date", tool="exec"),
        Req_Exec(path="/bin/id", tool="exec"),
    ]

    for cmd in must:
        result = dispatch(vm, cmd)
        formatted = _format_result(cmd, result)
        print(f"{CLI_GREEN}AUTO{CLI_CLR}: {formatted}")
        log.append({"role": "user", "content": formatted})

    log.append({"role": "user", "content": task_text})

    for i in range(30):
        step = f"step_{i + 1}"
        started = time.time()
        job = query_next_step_json(
            client,
            model,
            log,
            trace_dir,
            trace_prefix,
            i + 1,
        )
        elapsed_ms = int((time.time() - started) * 1000)

        print(
            f"Next {step}... {job.plan_remaining_steps_brief[0]} ({elapsed_ms} ms)\n"
            f"  {job.function}"
        )

        try:
            result = dispatch(vm, job.function)
            txt = _format_result(job.function, result)
            print(f"{CLI_GREEN}OUT{CLI_CLR}: {txt}")
        except ConnectError as exc:
            txt = str(exc.message)
            print(f"{CLI_RED}ERR {exc.code}: {exc.message}{CLI_CLR}")

        if isinstance(job.function, ReportTaskCompletion):
            status = CLI_GREEN if job.function.outcome == "OUTCOME_OK" else CLI_YELLOW
            print(f"{status}agent {job.function.outcome}{CLI_CLR}. Summary:")
            for item in job.function.completed_steps_laconic:
                print(f"- {item}")
            print(f"\n{CLI_BLUE}AGENT SUMMARY: {job.function.message}{CLI_CLR}")
            if job.function.grounding_refs:
                for ref in job.function.grounding_refs:
                    print(f"- {CLI_BLUE}{ref}{CLI_CLR}")
            break

        log.append({"role": "user", "content": _format_history_step(step, job, txt)})
