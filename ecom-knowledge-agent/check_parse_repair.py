"""Deterministic regression check for Phase-0 parse-layer repair.

Replays every historically-captured NextStep parse/schema failure (from the
trace JSONs under runs/traces/) through `_coerce_next_step` + NextStep
validation, with no model calls. Classifies each failed attempt as:

  REPAIRED              coercion produced a dict that now validates
  FALLS_THROUGH         coercion returns None (unparseable JSON, e.g. t45) ->
                        correctly handed to the existing retry path
  STILL_BROKEN          coercion produced a dict that STILL fails validation ->
                        a real remaining Phase-0 gap

Exit code 0 iff there are no STILL_BROKEN cases.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from pydantic import ValidationError

from agent import _coerce_next_step
from api_tools import NextStep

TRACES = Path(__file__).parent / "runs" / "traces"


def classify(candidate: str) -> str:
    coerced = _coerce_next_step(candidate)
    if coerced is None:
        # mirror the agent's fallback path
        try:
            NextStep.model_validate_json(candidate)
            return "REPAIRED"  # validates without coercion (shouldn't reach here for a failure)
        except ValidationError:
            return "FALLS_THROUGH"
    try:
        NextStep.model_validate(coerced)
        return "REPAIRED"
    except ValidationError:
        return "STILL_BROKEN"


def main() -> int:
    counts: Counter[str] = Counter()
    still_broken: list[tuple[str, str]] = []
    failures_seen = 0

    for trace in sorted(TRACES.glob("*.json")):
        try:
            data = json.loads(trace.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        for attempt in data.get("attempts", []):
            if "validation_error" not in attempt:
                continue  # only replay attempts that actually failed
            candidate = attempt.get("json_candidate")
            if candidate is None:
                continue
            failures_seen += 1
            verdict = classify(candidate)
            counts[verdict] += 1
            if verdict == "STILL_BROKEN":
                still_broken.append((trace.name, attempt.get("validation_error", "")[:200]))

    print(f"Replayed {failures_seen} captured parse/schema failures from {TRACES}")
    for k in ("REPAIRED", "FALLS_THROUGH", "STILL_BROKEN"):
        print(f"  {k:14} {counts.get(k, 0)}")

    if still_broken:
        print("\nSTILL_BROKEN (real remaining gaps):")
        for name, err in still_broken:
            print(f"  - {name}: {err}")
        return 1
    print("\nOK: no captured parse failure is left unhandled by Phase 0.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
