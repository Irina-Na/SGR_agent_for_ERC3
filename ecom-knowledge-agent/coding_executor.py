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

from answer_check import check_answer
from api_tools import Req_Exec, ReportTaskCompletion
from code_executor import ExecOutcome, run_script
from contract_validator import sanitize_grounding_refs, validate_report
from ecom_discovery import SessionDiscovery, discover
from ecom_runtime import EcomRuntime
from policy_collector import collect_task_policies, format_policy_collection_diag
from resolvers import repair_grounding_refs
from task_framing import (
    format_schema_diag,
    refresh_docs_for_trial,
    refresh_schema_for_trial,
)


# Phase B (Answer) — wrapper budget bumped from 60 → 100 after t15 exhausted
# at rt.sql() while iterating 6 products. The right architectural answer is for
# the model to batch (one IN/JOIN query instead of N per-row queries), but the
# prompt-side nudge alone won't fix it for every model; 100 is comfortable for
# 6-item lookups and still cheap enough to bound a runaway loop.
MAX_SCRIPT_ATTEMPTS = 4    # script-error retries (sandbox / runtime / contract)
MAX_GATE_RETRIES = 3       # finalization-gate retries
SCRIPT_TIMEOUT_SEC = 30.0
SCRIPT_WRAPPER_BUDGET = 100

# Phase A (Inspect) — smaller budgets; this phase only prints observations.
MAX_INSPECT_ATTEMPTS = 2
INSPECT_TIMEOUT_SEC = 20.0
INSPECT_WRAPPER_BUDGET = 30

# Phase C (Check) — at most this many "revise" rounds before we submit anyway.
MAX_REVISE_ROUNDS = 2


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


def _build_inspect_prompt(base_system_prompt: str) -> str:
    """Phase-A system prompt: discovery only, no answer yet.

    The model writes ONE python script whose sole job is to PRINT what a
    downstream agent would need to compute the answer: schema reflections,
    sample rows, the relevant /docs text, file listings. We capture stdout and
    feed it into Phase B."""
    return f"""{base_system_prompt.rstrip()}

# Output contract (Inspect phase — DISCOVERY ONLY)

You are in the INSPECT phase. Your job is **not** to answer the task yet. Your
job is to print everything a downstream agent would need to answer it: the
real schema (table and column names rotate per run — query `sqlite_schema` and
`PRAGMA table_info` via `rt.sql_raw`, do NOT trust any names you saw in
training), sample rows, governing /docs text, file listings — whatever the
task points at.

Rules:
- ONE fenced ```python ... ``` block, no prose before or after.
- Do NOT set `message`, `grounding_refs`, `outcome`, or `completed_steps_laconic`.
  Those belong to the Answer phase, which is a separate call.
- Use `print(...)` liberally. Whatever you print goes verbatim into the next
  LLM call's context as your own observations. Print things you want
  yourself-in-the-next-turn to read.
- Wrapper budget: {INSPECT_WRAPPER_BUDGET}; wall-clock {INSPECT_TIMEOUT_SEC:.0f}s.
- Sandbox rules: no `open`/`eval`/`exec`, no dunder/private attribute access,
  no imports (json + re are preloaded). `rt` is NOT a module — do NOT `import rt`.

Runtime — the ONLY I/O surface is `rt.*` (these exact methods, nothing else exists):
  rt.read(path, number=False, start_line=0, end_line=0)  -> obj with .content
  rt.list(path="/")          -> obj with .entries  (each entry has .name, .kind)
  rt.tree(root="", level=2)  -> obj with .root
  rt.find(name, root="/", kind="all", limit=10)
  rt.search(pattern, root="/", limit=10)
  rt.sql(query, params=())    -> list[dict]  # ? or :name binds; values auto-quoted
  rt.sql_raw(query, params=()) -> str         # same binds; raw stdout
  rt.exec(path, args=(), stdin="")            # invoke a /bin/* utility
  rt.paths                                     # frozenset of /-rooted paths SEEN this trial
Do NOT invent methods (no `rt.list_tree`, no `rt.schema`, no `rt.query`). If a
helper you want doesn't appear above, compose the same effect from what does.

Inspect mode is tolerant of missing optional filesystem paths: if `rt.read`,
`rt.list`, or `rt.tree` targets a path that is absent in this trial, it returns an
empty object instead of aborting. Treat empty content/entries/children as "not
present" and continue with schema/SQL or the current `/docs` tree. Do not assume
doc-internal path mentions exist unless the current trial's tools confirm them.

Good inspect script shape:
  1. Reflect the schema with `rt.sql_raw("SELECT ... FROM sqlite_schema ...")`
     then `rt.sql_raw("PRAGMA table_info(<table>)")` for relevant tables.
     PRINT the raw output — do not parse silently.
  2. If the task names a category/kind/type, query the taxonomy table and
     PRINT its DISTINCT values so the next phase knows the exact spelling.
  3. If a governing /docs policy was named in framing, `rt.read(...)` it and
     PRINT the relevant excerpt.
  4. Stop. Do not compute the answer yet.

Worked example (task wording would name a category like "Tool Box and Bag"):
```python
# 1. real tables / columns this run
print("== schema ==")
print(rt.sql_raw("SELECT name, sql FROM sqlite_schema WHERE type='table';"))

# 2. taxonomy: what are 'Tool Box and Bag' rows actually called?
print("== product_kinds matching 'tool box' ==")
print(rt.sql_raw(
    "SELECT * FROM product_kinds WHERE LOWER(name) LIKE ? LIMIT 5",
    ("%tool box%",),
))

# 3. one sample row from the variants table so Answer knows the columns
print("== sample variant ==")
print(rt.sql_raw("SELECT * FROM product_variants LIMIT 1"))
```

**Do NOT emit native tool calls** (no `<|tool_calls_section_begin|>`). The
`rt.*` calls are plain Python inside the fenced block."""


def _build_coding_prompt(base_system_prompt: str) -> str:
    return f"""{base_system_prompt.rstrip()}

# Output contract (coding mode) — READ CAREFULLY

You solve the WHOLE task with ONE complete Python script. **This is NOT a REPL
and NOT a step loop.** You will NOT see any `print` output between scripts; the
script runs once in a sandbox and we read four required variables from its
final state. Do not write exploratory one-liners — write the full solution:
explore inside the script (with normal Python control flow), compute the
answer, and assign it before the script ends.

**Inspect before you commit to a parser/assumption.** A single complete script
that handles unfamiliar data should follow this shape — all inside the one
script, never across attempts:

  1. READ the raw data (file, SQL result, etc.) into a variable.
  2. INSPECT it in code: split into lines, find landmark labels, count tokens,
     check what currency/separator/columns are actually present. Do NOT assume
     a format you have not just verified from the actual bytes.
  3. CHOOSE a parse strategy from what you observed (multiple landmark
     strategies + try/except for fragile formats — currency labels can be
     `EUR `, `€`, `EUR\t`, etc.; separators vary; numbers may use `,` or `.`).
  4. PARSE and VERIFY (e.g. sum line-items, compare to a printed total; if
     they don't reconcile, your parser is wrong — fall back to another
     strategy or read more landmarks).
  5. Only then COMPUTE the answer and assign `message`.

Brittle assumptions made before inspection are the #1 cause of wrong answers
in coding mode. A script that committed to one regex without looking at the
data first will fail silently on format variants — handle that inside the
script with verification + fallbacks, not by waiting for a retry.

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
  rt.sql(query, params=())    -> list[dict]  # ? or :name binds; values auto-quoted
  rt.sql_raw(query, params=()) -> str         # same binds; raw stdout
  rt.exec(path, args=(), stdin="")            # invoke a /bin/* utility
  rt.paths                                     # frozenset of /-rooted paths SEEN this trial

**rt.sql() returns dict rows where EVERY value is a string** (CSV parse — no
type coercion). Cast before numeric comparison: `int(row["qty"]) >= 1`, not
`row["qty"] >= 1`. Forgetting this raises `TypeError: '>=' not supported
between instances of 'str' and 'int'`.

**Batch queries — do NOT loop one rt.sql() per item.** Wrapper budget is
{SCRIPT_WRAPPER_BUDGET} total across the whole script; a per-item loop over
6 products easily exhausts it. Prefer ONE query with `WHERE col IN (...)` /
`JOIN ... ON ...` / a CTE, then group the results in Python. If you absolutely
must iterate, do it over the result set you already fetched, not by issuing a
fresh `rt.sql()` per item.

**grounding_refs are paths your tools actually returned this trial.** Tables
are NOT paths — `/proc/catalog`, `/proc/catalog/product_variants` are table
names, not citeable files. The right cite is the per-row `record_path` (or
similar) column value, which `rt.sql()` returns alongside the data. Likewise,
cite a /docs file only if you actually called `rt.read(...)` on it this trial.

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
in a single pass.

Worked example — answers "How many products are Tool Box and Bag? Format: <COUNT:%d>":
```python
# ONE batched query: join taxonomy + variants, get the count AND a path per row
rows = rt.sql(
    "SELECT pv.record_path FROM product_variants pv "
    "JOIN product_kinds pk ON pv.product_kind_id = pk.id "
    "WHERE pk.name = ?",
    ("Tool Box and Bag",),
)
count = len(rows)
# cite a handful of real /-rooted paths the query returned (NOT 'product_variants')
sample_paths = [r["record_path"] for r in rows[:3] if r.get("record_path")]

message = f"<COUNT:{{count}}>"
grounding_refs = sample_paths
outcome = "OUTCOME_OK"
completed_steps_laconic = [
    "Joined product_variants to product_kinds by name",
    "Counted matching rows",
    "Cited sample record_path values",
]
```

Worked example — inventory question with numeric cast + null guard:
```python
# ONE query that pre-filters to the store + products of interest
rows = rt.sql(
    "SELECT pv.record_path, si.quantity FROM store_inventory si "
    "JOIN product_variants pv ON pv.id = si.variant_id "
    "JOIN stores s ON s.id = si.store_id "
    "WHERE s.name = ? AND pv.sku IN (?, ?, ?)",
    ("West Vienna PowerTool", "SKU-A", "SKU-B", "SKU-C"),
)
# CSV cells are strings; row.get(...) can be None when the join misses a row.
qty = sum(1 for r in rows if int(r.get("quantity") or 0) >= 3)

message = f"qty={{qty}}"
grounding_refs = [r["record_path"] for r in rows if r.get("record_path")]
outcome = "OUTCOME_OK"
completed_steps_laconic = ["batched inventory query", "counted >=3 stock"]
```"""


def _query_script(
    client,
    model: str,
    messages: list[dict],
    trace_dir: Path | None,
    trace_prefix: str,
    attempt: int,
    phase: str = "script",
) -> tuple[str, str, str, int]:
    """Single LLM call → (raw_content, extracted_script, extraction_note, elapsed_ms).

    `phase` is the trace artifact stem: "script" for the Answer phase (legacy),
    "inspect" for the discovery-coverage phase. Lets a single helper serve both
    phases without coupling either to the executor's loop shape."""
    started = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_completion_tokens=32768,
    )
    elapsed_ms = int((time.time() - started) * 1000)
    content = resp.choices[0].message.content or ""
    script, note = _extract_python_block(content)
    if trace_dir is not None:
        trace_dir.mkdir(parents=True, exist_ok=True)
        (trace_dir / f"{trace_prefix}_{phase}_{attempt:02d}.json").write_text(
            json.dumps(
                {"attempt": attempt, "phase": phase, "elapsed_ms": elapsed_ms,
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

    # Build BOTH system prompts up front. The Inspect prompt forbids the
    # answer-contract variables; the Answer prompt requires them. They share
    # the same base (build_system_prompt) so /docs and the schema snapshot
    # are present in both phases.
    base_system = build_system_prompt(discovery)
    inspect_system = _build_inspect_prompt(base_system)
    answer_system = _build_coding_prompt(base_system)

    # Shared user-side preamble that both phase logs reuse: bootstrap (time +
    # identity) and the task itself. Framing/schema deltas append below.
    preamble: list[dict] = []

    # Bootstrap (identical to the step loop): time + identity per trial.
    must: list[Req_Exec] = []
    if discovery.time_tool:
        must.append(Req_Exec(tool="exec", path=f"/bin/{discovery.time_tool}"))
    if discovery.identity_tool:
        must.append(Req_Exec(tool="exec", path=f"/bin/{discovery.identity_tool}"))
    runtime_context: list[str] = []
    for cmd in must:
        try:
            result = rt.execute(cmd)
            formatted = _format_result(cmd, result)
            print(f"{CLI_GREEN}AUTO{CLI_CLR}: {formatted}")
            preamble.append({"role": "user", "content": formatted})
            runtime_context.append(formatted)
        except ConnectError as exc:
            print(f"{CLI_YELLOW}AUTO {cmd.path} failed (continuing): {exc.message}{CLI_CLR}")

    preamble.append({"role": "user", "content": task_text})

    # Policy collection pre-step. The injected context lives in `preamble`, so
    # both Inspect and Answer phases see the same runtime-selected /docs content.
    trial_discovery = refresh_docs_for_trial(rt, discovery)
    drift = len(trial_discovery.docs_tree) - len(discovery.docs_tree)
    collection = collect_task_policies(
        task_text,
        trial_discovery,
        rt,
        client,
        model,
        runtime_context="\n\n".join(runtime_context),
    )
    if collection.injected_context:
        selected_count = len(collection.authoritative_docs) + len(collection.candidate_docs)
        print(f"{CLI_BLUE}POLICY: injected {selected_count} runtime-selected docs{CLI_CLR}")
        preamble.append({"role": "user", "content": collection.injected_context})
    print(format_policy_collection_diag(collection, len(trial_discovery.docs_tree), drift))

    # Trial-scoped schema refresh — schema is in the base system prompt from
    # the run-scoped baseline, so we only inject a separate user message when
    # the trial dump differs materially (otherwise we'd just duplicate text
    # the model already has).
    trial_discovery = refresh_schema_for_trial(rt, trial_discovery)
    baseline_schema = discovery.schema_snapshot or ""
    trial_schema = trial_discovery.schema_snapshot or ""
    print(format_schema_diag(len(baseline_schema), len(trial_schema)))
    if trial_schema and trial_schema != baseline_schema:
        preamble.append({"role": "user", "content": (
            "# REFRESHED SQL SCHEMA (this trial — overrides the baseline in "
            "the system prompt):\n" + trial_schema
        )})

    # ---- Phase A: Inspect (discovery-coverage gate before any actions) ----
    inspect_log: list[dict] = [{"role": "system", "content": inspect_system}, *preamble]
    inspect_stdout = ""
    inspect_script_source = ""
    for attempt in range(1, MAX_INSPECT_ATTEMPTS + 1):
        print(f"{CLI_BLUE}--- inspect attempt {attempt}/{MAX_INSPECT_ATTEMPTS} ---{CLI_CLR}")
        try:
            content, script, note, elapsed_ms = _query_script(
                client, model, inspect_log, trace_dir, trace_prefix, attempt,
                phase="inspect",
            )
        except Exception as exc:  # noqa: BLE001
            print(f"{CLI_RED}LLM call failed: {exc}{CLI_CLR}")
            inspect_log.append({"role": "user", "content":
                       f"The previous LLM call failed: {exc}. Retry with one fenced python block."})
            continue
        print(f"inspect script ({elapsed_ms} ms, {len(script)} chars; raw {len(content)} chars"
              f"{'; note: ' + note[:60] if note else ''})")

        if not content.strip() or not script.strip():
            inspect_log.append({"role": "user", "content": (
                "Your previous response had no usable python block. Reply with "
                "EXACTLY one ```python ... ``` block that uses rt.* and print() "
                "to dump schema/sample-rows/doc-text. No prose, no preamble."
            )})
            continue

        outcome = run_script(
            script, rt,
            timeout_sec=INSPECT_TIMEOUT_SEC,
            wrapper_budget=INSPECT_WRAPPER_BUDGET,
            tolerate_not_found=True,
        )
        if not outcome.ok:
            print(f"{CLI_YELLOW}inspect error: {outcome.error}{CLI_CLR}")
            inspect_log.append({"role": "user", "content": _script_error_feedback(outcome)})
            continue
        if not outcome.captured_stdout.strip():
            print(f"{CLI_YELLOW}inspect produced no stdout — re-prompting{CLI_CLR}")
            inspect_log.append({"role": "user", "content": (
                "Your inspect script ran but printed nothing. The next phase needs "
                "to read your observations — call print() with the schema reflection "
                "and any sample rows / doc text the task hinges on."
            )})
            continue

        inspect_stdout = outcome.captured_stdout
        inspect_script_source = script
        print(f"{CLI_GREEN}inspect captured {len(inspect_stdout)} chars of stdout{CLI_CLR}")
        break
    else:
        # Pre-action gate triggered — abort the trial without submitting.
        print(f"{CLI_RED}coding executor: inspect phase failed; aborting trial without submit{CLI_CLR}")
        return

    # ---- Phase B + C: Answer with Check gate (revise loop) ----
    answer_log: list[dict] = [
        {"role": "system", "content": answer_system},
        *preamble,
        {"role": "user", "content": (
            "# OBSERVATIONS from your inspect script (your own stdout — use "
            "these exact names/values, do not guess):\n" + inspect_stdout
        )},
    ]

    gate_retries = 0
    revise_rounds = 0
    last_script_source = ""
    # Best candidate we've built across attempts. Used as the fallback submit if
    # the loop exhausts without ever reaching the clean-accept path — better to
    # send a flawed answer than score 0 by forfeit (a later script-error attempt
    # doesn't erase an earlier viable report).
    last_report: Optional[ReportTaskCompletion] = None
    for attempt in range(1, MAX_SCRIPT_ATTEMPTS + 1):
        print(f"{CLI_BLUE}--- answer attempt {attempt}/{MAX_SCRIPT_ATTEMPTS} "
              f"(revises {revise_rounds}/{MAX_REVISE_ROUNDS}) ---{CLI_CLR}")
        try:
            content, script, note, elapsed_ms = _query_script(
                client, model, answer_log, trace_dir, trace_prefix, attempt,
                phase="script",
            )
        except Exception as exc:  # noqa: BLE001
            print(f"{CLI_RED}LLM call failed: {exc}{CLI_CLR}")
            answer_log.append({"role": "user", "content":
                       f"The previous LLM call failed: {exc}. Retry with one fenced python block."})
            continue
        print(f"script ({elapsed_ms} ms, {len(script)} chars; raw {len(content)} chars"
              f"{'; note: ' + note[:60] if note else ''})")

        if not content.strip():
            answer_log.append({"role": "user", "content": (
                "Your previous response was EMPTY (0 tokens of output). This usually "
                "means the model spent its budget on reasoning. Reply with ONLY one "
                "concise fenced python block — no prose, no preamble — that solves "
                "the task using `rt.*` and assigns `message`, `grounding_refs`, "
                "`outcome`, `completed_steps_laconic` at the end."
            )})
            continue
        if not script.strip():
            answer_log.append({"role": "user", "content": (
                f"Your previous response contained no usable python script "
                f"({note or 'no fenced block found'}). Reply now with EXACTLY one "
                "fenced python block — start with ```python on its own line, end "
                "with ```. No prose before or after."
            )})
            continue

        last_script_source = script
        outcome = run_script(
            script, rt,
            timeout_sec=SCRIPT_TIMEOUT_SEC,
            wrapper_budget=SCRIPT_WRAPPER_BUDGET,
        )
        if not outcome.ok:
            print(f"{CLI_YELLOW}script error: {outcome.error}{CLI_CLR}")
            answer_log.append({"role": "user", "content": _script_error_feedback(outcome)})
            continue

        ns = outcome.namespace
        report = _build_report(ns)
        if report is None or not (report.message or "").strip():
            print(f"{CLI_YELLOW}script output did not satisfy ReportTaskCompletion contract"
                  f"{' (empty message)' if report is not None else ''}{CLI_CLR}")
            feedback = _contract_feedback(ns) if report is None else (
                "Your previous script executed but left `message` empty. Set "
                "`message`, `grounding_refs`, `outcome`, `completed_steps_laconic` "
                "before the script ends."
            )
            answer_log.append({"role": "user", "content": feedback})
            continue

        for correction in repair_grounding_refs(rt, discovery, report):
            print(f"{CLI_BLUE}{correction}{CLI_CLR}")

        # Snapshot every viable report — covers us if a later attempt crashes
        # in the sandbox or the validation gate keeps rejecting until budget runs out.
        last_report = report

        violations = validate_report(report, rt.paths, discovery, rt.docs_read)
        if violations and gate_retries < MAX_GATE_RETRIES and attempt < MAX_SCRIPT_ATTEMPTS:
            gate_retries += 1
            msg = "VALIDATION_FAILED (do not resubmit unchanged):\n- " + "\n- ".join(violations)
            print(f"{CLI_YELLOW}{msg}{CLI_CLR}")
            answer_log.append({"role": "user", "content":
                       msg + "\nRevise the script so its `message`/`grounding_refs`/"
                       "`outcome` satisfy the gate, and return one fenced python block."})
            continue

        # Phase C — discovery-coverage gate before submit.
        verdict = check_answer(
            task_text=task_text,
            inspect_stdout=inspect_stdout,
            script_source=last_script_source,
            report=report,
            client=client,
            model=model,
            trace_dir=trace_dir,
            trace_prefix=trace_prefix,
            attempt=attempt,
        )
        print(f"{CLI_BLUE}CHECK: verdict={verdict.verdict} confidence={verdict.confidence:.2f} "
              f"critique={verdict.critique[:120]!r}{CLI_CLR}")

        if verdict.verdict == "revise" and revise_rounds < MAX_REVISE_ROUNDS \
                and attempt < MAX_SCRIPT_ATTEMPTS:
            revise_rounds += 1
            answer_log.append({"role": "user", "content": (
                "CHECK FAILED — do not resubmit unchanged. Critique:\n"
                f"{verdict.critique}\n"
                "Revise the script using the OBSERVATIONS above (real schema, "
                "real distinct values) and return one fenced python block."
            )})
            continue

        if verdict.verdict == "abort":
            print(f"{CLI_YELLOW}CHECK aborted: {verdict.critique}{CLI_CLR}")
            try:
                report = ReportTaskCompletion(
                    tool="report_completion",
                    message=f"(check aborted) {verdict.critique}".strip(),
                    grounding_refs=list(report.grounding_refs),
                    outcome="OUTCOME_NONE_UNSUPPORTED",
                    completed_steps_laconic=list(report.completed_steps_laconic),
                )
            except (ValidationError, TypeError, ValueError):
                pass  # fall back to the model's report if synthesizing fails

        # accept (or revise budget exhausted, or abort with synthetic report) — submit.
        report, dropped = sanitize_grounding_refs(report, rt.paths, discovery)
        if dropped:
            print(f"{CLI_YELLOW}sanitize: dropped non-path grounding_refs: {dropped}{CLI_CLR}")
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

    if last_report is not None:
        print(f"{CLI_RED}coding executor: exhausted retries; submitting last viable "
              f"candidate rather than forfeiting the trial{CLI_CLR}")
        last_report, dropped = sanitize_grounding_refs(last_report, rt.paths, discovery)
        if dropped:
            print(f"{CLI_YELLOW}sanitize: dropped non-path grounding_refs: {dropped}{CLI_CLR}")
        try:
            rt.execute(last_report)
        except ConnectError as exc:
            print(f"{CLI_RED}ERR {exc.code}: {exc.message}{CLI_CLR}")
        print(f"{CLI_YELLOW}agent {last_report.outcome} (fallback submit){CLI_CLR}")
        print(f"{CLI_BLUE}AGENT SUMMARY: {last_report.message}{CLI_CLR}")
        for ref in last_report.grounding_refs:
            print(f"- {CLI_BLUE}{ref}{CLI_CLR}")
        return

    print(f"{CLI_RED}coding executor: exhausted retries with no viable candidate to submit{CLI_CLR}")
