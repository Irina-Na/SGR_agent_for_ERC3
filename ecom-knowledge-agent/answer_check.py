"""Phase C of coding mode — discovery-coverage gate before submit.

After the Answer script produces a candidate ReportTaskCompletion, we make ONE
more LLM call here to judge whether the answer is plausibly grounded in what
the Inspect phase actually observed. The motivating failure: a script
hard-codes a guessed table/column, gets back zero rows, and submits a
confident-but-wrong `<count: 0>`. Inspect captured the real schema; Check sees
both and can say "the script never touched that schema — revise."

Three outcomes:
    accept  → submit unchanged
    revise  → append critique to the log, loop back to Phase B with one more attempt
    abort   → submit OUTCOME_NONE_UNSUPPORTED with the critique as `message`

Fail-soft: if the LLM call errors or returns un-parseable JSON, default to
`accept` — Check must never block on its own infrastructure failure, otherwise
the gate is worse than no gate.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Literal, Optional

from openai import OpenAI
from pydantic import BaseModel, Field

from api_tools import ReportTaskCompletion


class CheckVerdict(BaseModel):
    verdict: Literal["accept", "revise", "abort"] = "accept"
    critique: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


_SYSTEM = """You are reviewing a candidate answer that another agent produced inside a sandboxed environment. You are NOT writing code yourself; you only judge whether the candidate is consistent with what the agent actually observed.

You will see:
  1. The task the agent was given.
  2. The OBSERVATIONS the agent printed during an inspection phase (real tool output: schema reflections, sample rows, doc text, file listings).
  3. The PYTHON SCRIPT the agent then wrote to compute its answer.
  4. The CANDIDATE answer the script produced (message + grounding_refs + outcome).

Decide ONE of:
  - "accept": the answer is plausibly grounded in the observations. Submit unchanged.
  - "revise": the script ignored or contradicted the observations (most common: hardcoded a table/column/value not present in the observed schema; computed zero/empty when the observations show data exists; cited a path that never appears in the observations). Provide a short, concrete critique the next attempt can act on.
  - "abort": the environment genuinely cannot answer this task (no relevant data, no governing policy). Rare.

Bias toward "accept" when you cannot find a specific defect — a wrong "revise" wastes an attempt. Bias toward "revise" when the candidate answer is `0`, empty, or "unknown" AND the observations show non-trivial data that the script did not consult. Never invent observations the agent did not print.

Output rules — read carefully, models routinely get this wrong:
- Return ONE JSON OBJECT that is an INSTANCE of the schema below — NOT the schema itself.
- Do NOT include keys named "properties", "title", "type", or "$defs". Those belong to the schema definition, not your answer.
- The top-level keys of your reply must be exactly: "verdict", "critique", "confidence".
- No markdown, no code fences, no prose before or after the JSON.

Example of a correct reply (shape only — your verdict/critique/confidence should reflect the actual review):
{"verdict":"revise","critique":"Script queried table `product` but observations show the real table is `product_variants`; the COUNT was zero because the table doesn't exist.","confidence":0.9}

Schema your reply must validate against (this is the contract, not a template to copy):
"""


_INSPECT_STDOUT_CAP_FOR_CHECK = 6 * 1024  # chars of inspect-stdout we forward to Check


def _extract_json(content: str) -> str:
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
    s, e = text.find("{"), text.rfind("}")
    return text[s:e + 1].strip() if (s != -1 and e != -1 and s < e) else text


def _unwrap_schema_echo(raw: str) -> str:
    """Defensive: if the model echoed back the JSON Schema (with `properties`,
    `title`, `type`) instead of an instance, lift the `properties` payload
    so Pydantic sees a valid instance. Common Qwen/Kimi failure mode.

    No-op when the input is already a clean instance."""
    if not raw:
        return raw
    try:
        obj = json.loads(raw)
    except Exception:  # noqa: BLE001
        return raw
    if not isinstance(obj, dict):
        return raw
    # The schema-echo pattern: top-level dict contains "properties" (and usually
    # "title"/"type") but lacks our actual fields. Promote properties up.
    if "properties" in obj and isinstance(obj["properties"], dict) \
            and "verdict" not in obj:
        return json.dumps(obj["properties"])
    return raw


def check_answer(
    task_text: str,
    inspect_stdout: str,
    script_source: str,
    report: ReportTaskCompletion,
    client: OpenAI,
    model: str,
    trace_dir: Optional[Path] = None,
    trace_prefix: str = "ecom",
    attempt: int = 1,
) -> CheckVerdict:
    """One LLM call → verdict. Fail-soft to `accept` on any error."""
    schema = json.dumps(CheckVerdict.model_json_schema(), ensure_ascii=False)
    # Keep the Check input bounded — t14's inspect dumped 25KB which truncated
    # the response under the old 2048 token cap. Schema landmarks are what the
    # gate actually needs; 6KB is plenty.
    bounded_stdout = inspect_stdout or "(empty)"
    if len(bounded_stdout) > _INSPECT_STDOUT_CAP_FOR_CHECK:
        bounded_stdout = (
            bounded_stdout[:_INSPECT_STDOUT_CAP_FOR_CHECK]
            + f"\n... (truncated from {len(inspect_stdout)} chars)"
        )
    user = (
        f"# Task\n{task_text}\n\n"
        f"# OBSERVATIONS (Inspect-phase stdout)\n{bounded_stdout}\n\n"
        f"# Script the agent wrote\n```python\n{script_source}\n```\n\n"
        f"# Candidate answer\n"
        f"message: {report.message!r}\n"
        f"grounding_refs: {list(report.grounding_refs)}\n"
        f"outcome: {report.outcome}\n"
    )
    started = time.time()
    raw = ""
    parse_error: Optional[str] = None
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM + schema},
                {"role": "user", "content": user},
            ],
            max_completion_tokens=4096,
        )
        raw = resp.choices[0].message.content or ""
        extracted = _unwrap_schema_echo(_extract_json(raw))
        verdict = CheckVerdict.model_validate_json(extracted)
        # Schema-echo defense — if pydantic accepted defaults because the model
        # gave us no real fields, the resulting empty-critique + 0.5-confidence
        # pair is meaningless. Treat it as a parse failure so we don't silently
        # rubber-stamp; the loop will downgrade to revise (see below).
        if not verdict.critique.strip() and verdict.confidence == 0.5 \
                and verdict.verdict == "accept":
            parse_error = "empty verdict (defaults only)"
    except Exception as exc:  # noqa: BLE001
        parse_error = str(exc)
        verdict = CheckVerdict(
            verdict="accept", critique=f"(check failed: {exc})", confidence=0.0,
        )

    # If Check infrastructure failed (parse error, defaults-only echo), bias to
    # revise rather than blindly accepting — the gate exists exactly to catch
    # confidently-wrong submissions, and rubber-stamping defeats it. We only
    # downgrade to revise (not abort): the next attempt may produce a verdict
    # we can actually read.
    if parse_error is not None:
        verdict = CheckVerdict(
            verdict="revise",
            critique=(
                f"Check failed to produce a usable verdict ({parse_error}). "
                "Re-emit the answer script, double-checking that every "
                "table/column/value comes from the OBSERVATIONS above."
            ),
            confidence=0.0,
        )
    elapsed_ms = int((time.time() - started) * 1000)

    if trace_dir is not None:
        try:
            trace_dir.mkdir(parents=True, exist_ok=True)
            (trace_dir / f"{trace_prefix}_check_{attempt:02d}.json").write_text(
                json.dumps(
                    {
                        "attempt": attempt,
                        "elapsed_ms": elapsed_ms,
                        "verdict": verdict.model_dump(),
                        "raw": raw,
                    },
                    ensure_ascii=False, indent=2,
                ),
                encoding="utf-8",
            )
        except Exception:  # noqa: BLE001 — tracing must never break the run
            pass
    return verdict
