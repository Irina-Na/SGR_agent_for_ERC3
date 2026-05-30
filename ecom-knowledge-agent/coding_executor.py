"""Phase-3 coding sub-executor — drops into the existing Framing/Resolver/Gate
wrapping, replacing the step loop with a single-script LLM call.

Lifecycle per task (mirrors run_agent's setup, swaps only the executor):
  1. bootstrap (time/identity AUTO calls, same as step loop)
  2. append task text
  3. **framing pre-step** (unchanged)  — governing /docs policy is surfaced into
     the message log just like the step loop
  4. coding loop:
       a. one LLM call → script source (fenced python block)
       b. AST screen + sandbox exec via code_executor.run_script
       c. on error → feed back the error, bounded retries
       d. on success → build ReportTaskCompletion from the script's namespace
  5. **resolver** (unchanged) — repair bare-id grounding_refs
  6. **gate** (unchanged) — validate_report; on violation, bounded retries that
     re-prompt the script
  7. submit via rt.execute

The wrappers are the only I/O surface (Phase 1) and they already auto-capture
the evidence ledger, so a substituted/seen path enters `rt.paths` regardless of
whether the step loop or a sandboxed script ran it.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Literal, Optional

from connectrpc.errors import ConnectError
from pydantic import ValidationError

from bitgn.vm.ecom.ecom_connect import EcomRuntimeClientSync

from api_tools import Req_Exec, ReportTaskCompletion
from code_executor import ExecOutcome, run_script
from contract_validator import validate_report
from ecom_discovery import SessionDiscovery, discover
from ecom_runtime import EcomRuntime
from resolvers import repair_grounding_refs
from task_framing import format_framing_diag, frame_task, refresh_docs_for_trial


MAX_SCRIPT_ATTEMPTS = 4    # script-error retries (sandbox / runtime / contract)
MAX_GATE_RETRIES = 3       # finalization-gate retries (shares the same budget shape as the step loop)
SCRIPT_TIMEOUT_SEC = 30.0
SCRIPT_WRAPPER_BUDGET = 60


_NATIVE_TOOL_MARKERS = (
    "<|tool_calls_section_begin|>", "<|tool_call_begin|>",
    "<|tool_calls_section_end|>", "<|tool_call_end|>",
)


def _extract_python_block(content: str) -> tuple[str, str]:
    """Return (script, note).

    `note` is "" on a clean extraction; otherwise it's a short diagnostic the
    caller can include in the retry feedback (truncation, native tool calls).

    Handles four shapes Kimi actually emits:
    - clean fenced block (the happy case)
    - opening fence but no closing fence (response was truncated mid-stream)
    - bare native tool-call section ``<|tool_calls_section_begin|>`` (Kimi's
      instruction-following bug — that's NOT a script, return empty)
    - no fence at all → return content as-is (could still be a valid script)
    """
    if any(m in content for m in _NATIVE_TOOL_MARKERS):
        return "", ("emitted a native tool-call section; that is not a python "
                    "script. Reply with one ```python ... ``` fenced block.")
    # closed fence (preferred)
    m = re.search(r"```(?:python)?\s*\n?(.*?)```", content, re.DOTALL)
    if m:
        return m.group(1).strip(), ""
    # open-only fence (truncated response — strip the opening ```python so the
    # remaining code at least gets a chance to parse; warn the caller)
    m = re.search(r"```(?:python)?\s*\n(.*)$", content, re.DOTALL)
    if m:
        return m.group(1).strip(), (
            "your previous response was TRUNCATED — the opening ```python "
            "fence had no closing ```. The script likely ended mid-statement. "
            "Keep the next script tighter so it fits in one response."
        )
    return content.strip(), ""


def _build_coding_prompt(base_system_prompt: str) -> str:
    return f"""{base_system_prompt.rstrip()}

# Output contract (coding mode) — READ CAREFULLY

You solve the WHOLE task with ONE complete Python script. **This is NOT a REPL
and NOT a step loop.** You will NOT see any `print` output between scripts; the
script runs once in a sandbox and we read four required variables from its
final state. Do not write exploratory one-liners — write the full solution:
explore inside the script (with normal Python control flow), compute the
answer, and assign it before the script ends.

Return exactly one fenced Python block, no prose before or after:

```python
# your complete solution
```

**Do NOT emit native tool calls or tool-call sections** (no
`<|tool_calls_section_begin|>`, no `<|tool_call_begin|>`). The `rt.*` calls are
plain Python — written as method calls inside the fenced block, not as the
host's native tool-calling format.

Runtime — only I/O is `rt.*` (every call counts against a budget of {SCRIPT_WRAPPER_BUDGET}):
  rt.read(path, number=False, start_line=0, end_line=0)  -> obj with .content
  rt.list(path="/")          -> obj with .entries  (each entry has .name, .kind)
  rt.tree(root="", level=2)  -> obj with .root
  rt.find(name, root="/", kind="all", limit=10)
  rt.search(pattern, root="/", limit=10)
  rt.sql(query: str) -> list[dict]   # CSV-parsed rows; may be [] on non-CSV output
  rt.sql_raw(query: str) -> str       # raw stdout — format-agnostic, always safe
  rt.exec(path, args=(), stdin="")    # invoke a /bin/* utility
  rt.paths                            # frozenset of /-rooted paths the runtime
                                      # has SEEN in tool results this trial

PRE-IMPORTED modules — already available, **do NOT write `import` for these**:
  json   — `json.loads(...)`, `json.dumps(...)`
  re     — `re.search(...)`, `re.findall(...)`, etc.
No other imports are allowed and `import` is blocked by the sandbox.

Other sandbox rules:
- No `open`, no `eval`/`exec`/`compile`, no `setattr`/`delattr`.
- No dunder or single-underscore attribute access (no `.__class__`, no `._foo`).
- `getattr` and `hasattr` are allowed for normal attribute names.
- Wall-clock timeout: {SCRIPT_TIMEOUT_SEC:.0f}s. Exception or sandbox violation
  → we re-prompt with the error so you can fix the script.

Required — before the script ends, set these four top-level variables. If any
is missing or `message` is empty, we re-prompt:
  message: str                # final answer text matching the task's EXACT format
  grounding_refs: list[str]   # /-rooted paths returned by tools (never invent).
                              # Prefer values from `rt.paths`.
  outcome: str                # "OUTCOME_OK" / "OUTCOME_DENIED_SECURITY" /
                              # "OUTCOME_NONE_CLARIFICATION" /
                              # "OUTCOME_NONE_UNSUPPORTED" / "OUTCOME_ERR_INTERNAL"
  completed_steps_laconic: list[str]  # 3-6 short steps you took

All operating rules and grounding/policy/outcome discipline from the section
above apply unchanged. Write ONE complete script; explore, compute, and answer
in a single pass."""


def _query_script(
    client,
    model: str,
    messages: list[dict],
    trace_dir: Path | None,
    trace_prefix: str,
    attempt: int,
) -> tuple[str, str, str, int]:
    """Single LLM call → (raw_content, extracted_script, extraction_note, elapsed_ms)."""
    started = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_completion_tokens=16384,
    )
    elapsed_ms = int((time.time() - started) * 1000)
    content = resp.choices[0].message.content or ""
    script, note = _extract_python_block(content)
    if trace_dir is not None:
        trace_dir.mkdir(parents=True, exist_ok=True)
        (trace_dir / f"{trace_prefix}_script_{attempt:02d}.json").write_text(
            json.dumps(
                {"attempt": attempt, "elapsed_ms": elapsed_ms,
                 "content": content, "script": script, "note": note},
                ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )
    return content, script, note, elapsed_ms


def _build_report(ns: dict) -> Optional[ReportTaskCompletion]:
    try:
        return ReportTaskCompletion(
            tool="report_completion",
            message=str(ns.get("message", "")),
            grounding_refs=list(ns.get("grounding_refs", []) or []),
            outcome=str(ns.get("outcome", "OUTCOME_OK")),
            completed_steps_laconic=list(ns.get("completed_steps_laconic", []) or []),
        )
    except (ValidationError, TypeError, ValueError):
        return None


def _script_error_feedback(outcome: ExecOutcome) -> str:
    return (
        f"Your previous script failed: {outcome.error}\n"
        f"(wrappers used: {outcome.wrappers_used}/{SCRIPT_WRAPPER_BUDGET}; "
        f"timed_out: {outcome.timed_out})\n"
        "Fix it and return one corrected python fenced block."
    )


def _contract_feedback(ns: dict) -> str:
    missing = [k for k in ("message", "grounding_refs", "outcome",
                           "completed_steps_laconic") if k not in ns]
    return (
        "Your script executed but did not produce the required top-level "
        f"variables: missing {missing}. Re-emit the script and assign all four "
        "before it ends."
    )


def run_agent_coding(
    model: str,
    harness_url: str,
    task_text: str,
    provider: Literal["nebius", "openai"] = "nebius",
    discovery: Optional[SessionDiscovery] = None,
    trace_dir: Path | None = None,
    trace_prefix: str = "ecom",
) -> None:
    """Coding-mode executor. Same Framing/Resolver/Gate wrapping as the step
    loop; the step loop is replaced by a single-script LLM call per attempt."""
    # Lazy imports of agent.py helpers to avoid an import cycle (agent imports
    # this module via main.py routing).
    from agent import (
        get_llm_client,
        build_system_prompt,
        _format_result,
        CLI_BLUE, CLI_CLR, CLI_GREEN, CLI_RED, CLI_YELLOW,
    )

    client = get_llm_client(provider)
    vm = EcomRuntimeClientSync(harness_url)
    if discovery is None:
        print(f"{CLI_BLUE}Discovery (trial-scoped fallback)...{CLI_CLR}")
        discovery = discover(vm, client, model)

    rt = EcomRuntime(vm, sql_tool=discovery.sql_tool)

    system_prompt = _build_coding_prompt(build_system_prompt(discovery))
    log: list[dict] = [{"role": "system", "content": system_prompt}]

    # Bootstrap (identical to the step loop): time + identity per trial.
    must: list[Req_Exec] = []
    if discovery.time_tool:
        must.append(Req_Exec(tool="exec", path=f"/bin/{discovery.time_tool}"))
    if discovery.identity_tool:
        must.append(Req_Exec(tool="exec", path=f"/bin/{discovery.identity_tool}"))
    for cmd in must:
        try:
            result = rt.execute(cmd)
            formatted = _format_result(cmd, result)
            print(f"{CLI_GREEN}AUTO{CLI_CLR}: {formatted}")
            log.append({"role": "user", "content": formatted})
        except ConnectError as exc:
            print(f"{CLI_YELLOW}AUTO {cmd.path} failed (continuing): {exc.message}{CLI_CLR}")

    log.append({"role": "user", "content": task_text})

    # Framing pre-step — surface governing /docs policy into the log so the
    # script sees it on the first LLM call. Part A item 1: refresh /docs from
    # THIS trial (run-hoisted snapshot drifts across trials).
    trial_discovery = refresh_docs_for_trial(rt, discovery)
    drift = len(trial_discovery.docs_tree) - len(discovery.docs_tree)
    framing = frame_task(task_text, trial_discovery, client, model)
    paths = framing.governing_doc_paths
    if len(paths) == 1 and framing.confident:
        ladder = "authoritative"
        p = paths[0]
        try:
            content = rt.read(__import_req_read(p)).content  # see helper below
            print(f"{CLI_BLUE}FRAMING: governing policy {p}{CLI_CLR}")
            log.append({"role": "user", "content": (
                f"# GOVERNING POLICY for this task (authoritative — apply its "
                f"exact definition, grain, filters, and output format; it "
                f"overrides any default interpretation): {p}\n{content}"
            )})
        except ConnectError as exc:
            print(f"{CLI_YELLOW}FRAMING: could not read {p}: {exc.message}{CLI_CLR}")
            ladder = "none"
    elif paths:
        ladder = "candidates"
        print(f"{CLI_BLUE}FRAMING: {len(paths)} candidate policies (ambiguous scope){CLI_CLR}")
        listing = "\n".join(f"- {p}" for p in paths)
        log.append({"role": "user", "content": (
            "# CANDIDATE POLICIES — more than one may govern this task. "
            "Determine which scope (e.g. location/time/workflow) the task "
            "actually requires, then read and apply that one before answering:\n"
            + listing
        )})
    else:
        ladder = "none"
    # Part A item 2: always emit one diagnostic line (miss or hit).
    print(format_framing_diag(framing, len(trial_discovery.docs_tree), ladder, drift))

    gate_retries = 0
    for attempt in range(1, MAX_SCRIPT_ATTEMPTS + 1):
        print(f"{CLI_BLUE}--- coding attempt {attempt}/{MAX_SCRIPT_ATTEMPTS} ---{CLI_CLR}")
        try:
            content, script, note, elapsed_ms = _query_script(
                client, model, log, trace_dir, trace_prefix, attempt,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"{CLI_RED}LLM call failed: {exc}{CLI_CLR}")
            log.append({"role": "user", "content":
                       f"The previous LLM call failed: {exc}. Retry with one fenced python block."})
            continue
        print(f"script ({elapsed_ms} ms, {len(script)} chars; raw {len(content)} chars"
              f"{'; note: ' + note[:60] if note else ''})")

        # Empty-response guards — surfaced as a distinct, actionable failure so
        # the model isn't told "syntax error" when it actually produced nothing.
        if not content.strip():
            log.append({"role": "user", "content": (
                "Your previous response was EMPTY (0 tokens of output). This usually "
                "means the model spent its budget on reasoning. Reply with ONLY one "
                "concise fenced python block — no prose, no preamble — that solves "
                "the task using `rt.*` and assigns `message`, `grounding_refs`, "
                "`outcome`, `completed_steps_laconic` at the end."
            )})
            continue
        if not script.strip():
            log.append({"role": "user", "content": (
                f"Your previous response contained no usable python script "
                f"({note or 'no fenced block found'}). Reply now with EXACTLY one "
                "fenced python block — start with ```python on its own line, end "
                "with ```. No prose before or after."
            )})
            continue

        outcome = run_script(
            script, rt,
            timeout_sec=SCRIPT_TIMEOUT_SEC,
            wrapper_budget=SCRIPT_WRAPPER_BUDGET,
        )
        if not outcome.ok:
            print(f"{CLI_YELLOW}script error: {outcome.error}{CLI_CLR}")
            # Don't append the failed assistant turn — growing it inflates the
            # context and reasoning-budget models (Kimi) start emitting 0 chars.
            # The error message references what failed; that's enough.
            log.append({"role": "user", "content": _script_error_feedback(outcome)})
            continue

        ns = outcome.namespace
        report = _build_report(ns)
        if report is None or not (report.message or "").strip():
            print(f"{CLI_YELLOW}script output did not satisfy ReportTaskCompletion contract"
                  f"{' (empty message)' if report is not None else ''}{CLI_CLR}")
            feedback = _contract_feedback(ns) if report is None else (
                "Your previous script executed but left `message` empty. We did "
                "NOT submit. Write ONE complete script that explores AND answers "
                "in a single pass; do not write exploratory snippets expecting "
                "to see output between scripts. Set `message`, `grounding_refs`, "
                "`outcome`, `completed_steps_laconic` before the script ends."
            )
            log.append({"role": "user", "content": feedback})
            continue

        # Resolver (unchanged) — repair bare-id/unseen-path refs to real paths.
        for correction in repair_grounding_refs(rt, discovery, report):
            print(f"{CLI_BLUE}{correction}{CLI_CLR}")

        # Finalization gate (unchanged) — same budget discipline as step loop.
        violations = validate_report(report, rt.paths, discovery, rt.docs_read)
        if violations and gate_retries < MAX_GATE_RETRIES and attempt < MAX_SCRIPT_ATTEMPTS:
            gate_retries += 1
            msg = "VALIDATION_FAILED (do not resubmit unchanged):\n- " + "\n- ".join(violations)
            print(f"{CLI_YELLOW}{msg}{CLI_CLR}")
            log.append({"role": "user", "content":
                       msg + "\nRevise the script so its `message`/`grounding_refs`/"
                       "`outcome` satisfy the gate, and return one fenced python block."})
            continue

        # Accept and submit.
        try:
            rt.execute(report)
        except ConnectError as exc:
            print(f"{CLI_RED}ERR {exc.code}: {exc.message}{CLI_CLR}")
        status = CLI_GREEN if report.outcome == "OUTCOME_OK" else CLI_YELLOW
        print(f"{status}agent {report.outcome}{CLI_CLR}. Summary:")
        for item in report.completed_steps_laconic:
            print(f"- {item}")
        print(f"\n{CLI_BLUE}AGENT SUMMARY: {report.message}{CLI_CLR}")
        for ref in report.grounding_refs:
            print(f"- {CLI_BLUE}{ref}{CLI_CLR}")
        return

    print(f"{CLI_RED}coding executor: exhausted retries without a clean submit{CLI_CLR}")


def __import_req_read(path: str):
    """Late-binding factory so the framing read inside this module mirrors what
    agent.py does without an import cycle."""
    from api_tools import Req_Read
    return Req_Read(tool="read", path=path)
