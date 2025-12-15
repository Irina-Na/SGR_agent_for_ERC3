"""
Runtime security checker: applies policy constraints to a given intent/user/resource.

Use `policy_planner.py` at startup to fetch wiki, build a plan, and extract policies.
Then feed the extracted constraints (from policies.json) into `classify` for each request.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
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


def load_policy_constraints(policy_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load constraints map keyed by intent from a saved policies.json."""
    data = json.loads(Path(policy_path).read_text(encoding="utf-8"))
    raw = data.get("constraints_map") or data.get("constraints") or {}
    if isinstance(raw, list):
        return {c.get("intent"): c for c in raw if isinstance(c, dict) and c.get("intent")}
    if isinstance(raw, dict):
        return {k: v for k, v in raw.items() if isinstance(v, dict)}
    return {}


def classify(
    intent: str,
    user_ctx: dict,
    resource_ctx: dict,
    policy: dict | Dict[str, Dict[str, Any]],
    allow_on_missing: bool = False,
    on_missing=None,
) -> Decision:
    """
    Apply fail-closed classification of a request using a normalized policy dict or a constraints_map.

    Expected input (any of):
      - full policy dict from policies.json (with constraints/constraints_map)
      - already-extracted constraints_map { intent: { ..fields.. } }

    Constraint fields respected (all optional, default deny if missing):
      roles, override_roles, locations (['any'] to disable check),
      allow_if_exec, needs_responsibility, needs_assignment,
      needs_clarification, sensitivity (public|internal|sensitive|highly_sensitive).
    """
    constraints_map: Dict[str, Dict[str, Any]] = {}
    if isinstance(policy, dict):
        raw = policy.get("constraints") or policy.get("constraints_map") or policy
        if isinstance(raw, list):
            constraints_map = {c.get("intent"): c for c in raw if isinstance(c, dict) and c.get("intent")}
        elif isinstance(raw, dict):
            constraints_map = {k: v for k, v in raw.items() if isinstance(v, dict)}
    constraints = constraints_map.get(intent)
    if not constraints:
        if on_missing:
            return on_missing(intent, user_ctx, resource_ctx)
        if allow_on_missing:
            return Decision("allow", f"policy_missing_allowed:{intent}")
        return Decision("deny", f"policy_no_rule_for_{intent}")

    role = user_ctx.get("role")
    # Execs default to override unless explicitly disabled
    exec_override = role == "exec" and constraints.get("allow_if_exec", True)

    allowed_roles = constraints.get("roles", [])
    if allowed_roles and not (exec_override or role in allowed_roles):
        return Decision("deny", f"role_not_allowed:{role}")

    allowed_locations = constraints.get("locations")
    if allowed_locations and "any" not in allowed_locations and not exec_override:
        locs = {resource_ctx.get("project_location"), user_ctx.get("location")}
        if not any(loc in allowed_locations for loc in locs if loc):
            return Decision("deny", "location_scope_block")

    if constraints.get("needs_responsibility") and not exec_override:
        if not resource_ctx.get("is_owner_or_lead", False) and role not in constraints.get("override_roles", []):
            return Decision("deny", "not_responsible_for_project")

    if constraints.get("needs_assignment") and not exec_override:
        if not resource_ctx.get("user_on_project", False):
            return Decision("deny", "not_assigned_to_project")

    sensitivity = constraints.get("sensitivity")
    if sensitivity in ("sensitive", "highly_sensitive") and not exec_override:
        return Decision("deny", "sensitive_data_blocked")

    if constraints.get("needs_clarification") and not resource_ctx.get("target_resolved"):
        return Decision("clarify", "need_disambiguation")

    return Decision("allow", "ok")


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


def _build_llm_messages(intent: str, user_ctx: dict, resource_ctx: dict, policy_doc: dict) -> list[dict]:
    constraints_map = load_policy_constraints(Path(DEFAULT_POLICY_PATH))
    constraint = constraints_map.get(intent, {})
    policy_rules: Sequence[dict] = policy_doc.get("policies", [])
    relevant_rules = [r for r in policy_rules if intent in (r.get("intents") or [])]

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
        "request": intent,
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]


def llm_classify(
    intent: str,
    user_ctx: dict,
    resource_ctx: dict,
    policy_path: Path = DEFAULT_POLICY_PATH,
    model: str | None = None,
) -> Decision:
    """LLM-based classifier using the ideal manual policies as context."""
    if OpenAI is None:
        raise ImportError("openai package not installed; LLM classification unavailable")
    policy_doc = _load_full_policy(policy_path)
    messages = _build_llm_messages(intent, user_ctx, resource_ctx, policy_doc)
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


__all__ = ["Decision", "classify", "load_policy_constraints", "llm_classify"]
