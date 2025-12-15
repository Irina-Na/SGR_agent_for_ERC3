"""
Runtime security checker: LLM-based policy gate for a given user_query/user/resource.

Use `policy_planner.py` at startup to fetch wiki, build a plan, and extract policies;
route runtime checks through `llm_classify` (or its `classify` alias).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Sequence

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency for LLM mode
    OpenAI = None

try:
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover - optional dependency for LLM mode
    BaseModel = None  # type: ignore


@dataclass
class Decision:
    status: Literal["allow", "deny", "clarify"]
    reason: str


def _extract_constraints_map(policy: dict | Dict[str, Dict[str, Any]] | None) -> Dict[str, Dict[str, Any]]:
    """Normalize constraints map from a full policy dict or already-extracted map."""
    if not isinstance(policy, dict):
        return {}
    raw = policy.get("constraints_map") or policy.get("constraints") or policy
    if isinstance(raw, list):
        return {c.get("user_query"): c for c in raw if isinstance(c, dict) and c.get("user_query")}
    if isinstance(raw, dict):
        return {k: v for k, v in raw.items() if isinstance(v, dict)}
    return {}


def load_policy_constraints(policy_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load constraints map keyed by user_query from a saved policies.json."""
    data = json.loads(Path(policy_path).read_text(encoding="utf-8"))
    return _extract_constraints_map(data)


DEFAULT_POLICY_PATH = Path("agent_security_analyser/policy/plan-ideal-manual/policies.json")
DEFAULT_FAILURES_PATH = Path("agent_security_analyser/plans/security_policy_failures.json")
DATA_ROOT = Path("agent_security_analyser/data")
# who_am_i placeholder; replace with real call if available.
DEBUG_USER_CTX = {
    "user_id": "jonas_weiss",
    "role": "level_3",
    "location": "Munich",
}
# Shared resource context for the two failing tasks; tweak per task as needed.
DEBUG_RESOURCE_CTX = {
    "project_location": "Munich",
    "is_owner_or_lead": False,
    "user_on_project": False,
    "target_resolved": True,
}


class LlmDecision(BaseModel):  # type: ignore[misc]
    status: Literal["allow", "deny", "clarify"]
    reason: str = Field(..., description="short code or phrase")

    @classmethod
    def schema_str(cls) -> str:
        return json.dumps(cls.model_json_schema(), ensure_ascii=False, indent=2)


def _load_full_policy(policy_path: Path = DEFAULT_POLICY_PATH) -> dict:
    return json.loads(Path(policy_path).read_text(encoding="utf-8"))


def _build_llm_messages(user_query: str, user_ctx: dict, resource_ctx: dict, policy_doc: dict) -> list[dict]:
    constraints_map = _extract_constraints_map(policy_doc)
    constraint = constraints_map.get(user_query, {})
    policy_rules: Sequence[dict] = policy_doc.get("policies", []) or []
    relevant_rules = [r for r in policy_rules if user_query in (r.get("user_queries") or [])]

    system = (
        "You are a strict security decision service. "
        "Use ONLY the provided constraint and relevant_rules. "
        "If anything is missing or ambiguous, return deny. "
        "Output ONLY JSON matching the schema below; no prose. "
        "Schema:\n"
        f"{LlmDecision.schema_str()}"
    )
    user_payload = {
        "constraint": constraint,
        "relevant_rules": [
            {"name": r.get("name"), "summary": r.get("summary"), "allow": r.get("allow"), "deny": r.get("deny")}
            for r in relevant_rules
        ],
        "user_ctx": user_ctx,
        "resource_ctx": resource_ctx,
        "request": user_query,
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]


def llm_classify(
    user_query: str,
    user_ctx: dict,
    resource_ctx: dict,
    policy_path: Path = DEFAULT_POLICY_PATH,
    policy_doc: dict | None = None,
    model: str | None = None,
) -> Decision:
    """LLM-based classifier using the ideal manual policies as context."""
    if OpenAI is None:
        raise ImportError("openai package not installed; LLM classification unavailable")
    policy_doc = policy_doc or _load_full_policy(policy_path)
    messages = _build_llm_messages(user_query, user_ctx, resource_ctx, policy_doc)
    client = OpenAI()
    resp = client.chat.completions.create(
        model=model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=messages,
        temperature=0,
    )
    content = resp.choices[0].message.content or ""
    try:
        parsed = json.loads(content)
        result = LlmDecision.model_validate(parsed)
        return Decision(result.status, result.reason)
    except json.JSONDecodeError:
        pass
    except Exception:
        pass
    return Decision("deny", "llm_parse_error")


def classify(
    user_query: str,
    user_ctx: dict,
    resource_ctx: dict,
    policy: dict | Dict[str, Dict[str, Any]] | Path | str | None = None,
    allow_on_missing: bool = False,  # kept for API compatibility; ignored
    on_missing=None,  # kept for API compatibility; ignored
    model: str | None = None,
) -> Decision:
    """
    Legacy alias that routes classification exclusively through llm_classify.

    Any rule-based checks have been removed to avoid hard-coded outcomes; the
    LLM now decides based on the provided policy document.
    """
    policy_doc = policy if isinstance(policy, dict) else None
    if policy_doc is not None and "policies" not in policy_doc:
        # Allow passing a bare constraints_map and still feed it to the LLM.
        policy_doc = {"constraints_map": policy_doc}
    policy_path = Path(policy) if isinstance(policy, (str, Path)) else DEFAULT_POLICY_PATH
    return llm_classify(user_query, user_ctx, resource_ctx, policy_path=policy_path, policy_doc=policy_doc, model=model)


__all__ = ["Decision", "classify", "load_policy_constraints", "llm_classify"]
