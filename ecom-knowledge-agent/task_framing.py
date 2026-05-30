"""Task-framing pre-step (Phase 2.5).

Before the step loop runs, surface the `/docs` policy that GOVERNS this task's
answer — when one applies. The count/report bucket proved that the authoritative
semantics (grain, filters, output) live in planted policy docs, not in the schema:
e.g. a doc dictates `COUNT(DISTINCT sku)` for an open store in a named city. The
agent that reads the right-scoped doc passes; the one that reasons from the schema
alone fails. This step puts that doc in hand up front instead of hoping the agent
goes looking.

Matching discipline (see plan §2 count-bucket lever):
- SEMANTIC match, not literal — folder names, workflow verbs (count/counting/
  reporting), category and city vocab, and the file extension all rotate per the
  prime directive, so nothing here is hardcoded; we hand the discovered doc catalog
  + the task to the model and let it match on meaning.
- SCOPE-PRECISE + bidirectional: a doc narrowed to a scope (city/time/workflow) the
  task doesn't mention does NOT govern it; a task narrowed to a scope no doc covers
  has no governing doc. Prefer the single most specific match.
- REFUSE to auto-pick on ties: if ≥2 docs match equally, surface them as candidates
  for the agent to disambiguate — code never chooses the scope.
- PRECISION over recall: when unsure, return none. A wrong governing doc is worse
  than none (it yields a confidently-wrong answer).
- DEGRADE-safe: any failure → no injection → today's behavior. Never blocks.

Honest limitation: matching leans on discovered filename triggers, which degrade
when filenames are opaque hashes. A bounded, run-scoped content-read fallback is the
next increment; until then opaque-filename docs simply won't be surfaced (degrades to
today, never wrong).
"""
from __future__ import annotations

import json
from typing import List

from openai import OpenAI
from pydantic import BaseModel, Field

from ecom_discovery import SessionDiscovery


class FramingResult(BaseModel):
    governing_doc_paths: List[str] = Field(default_factory=list)
    confident: bool = False
    reason: str = ""


_SYSTEM = """You decide which discovered policy/reference document (if any) GOVERNS the answer to a specific task in this environment.

You are given the task and a catalog of available /docs files (path + inferred topic triggers). Some documents define the exact, authoritative rules for one operation on one subject at one scope — e.g. a specific metric or workflow, for a specific category of entity, sometimes narrowed to a specific location or time window. When such a document matches the task, its rules (grain, filters, output format) override any default interpretation, and the answer is not derivable without it.

Select the governing document(s) ONLY when BOTH hold:
- SUBJECT match: the document is about the same entity/category the task is about, AND
- SCOPE match (both directions): every qualifier the document is narrowed by — location, time window, workflow/metric type — is present in and consistent with the task. A document narrowed to a scope the task does not mention does NOT govern it. A task narrowed to a scope no document covers has no governing document.

Rules:
- Prefer the single MOST SPECIFIC match (an exactly-scoped doc beats a broader/unscoped one for a scoped task; an unscoped doc governs only an unscoped task).
- If two or more documents match the task equally and you cannot tell which scope the task requires, return ALL of them — do not guess.
- If no document governs, return an empty list.
- When unsure, prefer returning fewer (empty). A wrong governing document is worse than none.
- Match on meaning, not string patterns: folder, filename, workflow wording, and extensions vary.

Return only one JSON object, no prose/markdown, matching this schema:
"""


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


def frame_task(
    task_text: str,
    discovery: SessionDiscovery,
    llm_client: OpenAI,
    model: str,
) -> FramingResult:
    """Return the governing /docs policy path(s) for this task (possibly empty).

    Fails soft: on any error or empty catalog, returns an empty result so the caller
    simply proceeds with no injection (today's behavior)."""
    docs = list(discovery.docs_tree or [])
    if not docs:
        return FramingResult()

    catalog = [
        {"path": p, "triggers": discovery.policy_doc_index.get(p, [])}
        for p in docs
    ]
    schema = json.dumps(FramingResult.model_json_schema(), ensure_ascii=False)
    user = (
        f"# Task\n{task_text}\n\n"
        f"# Available /docs catalog (path + inferred triggers)\n"
        f"{json.dumps(catalog, ensure_ascii=False, indent=2)}\n"
    )
    try:
        resp = llm_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM + schema},
                {"role": "user", "content": user},
            ],
            max_completion_tokens=2048,
        )
        content = resp.choices[0].message.content or ""
        result = FramingResult.model_validate_json(_extract_json(content))
    except Exception:
        return FramingResult()

    # keep only paths that are actually in the discovered catalog (no hallucinated docs)
    known = set(docs)
    result.governing_doc_paths = [p for p in result.governing_doc_paths if p in known]
    return result
