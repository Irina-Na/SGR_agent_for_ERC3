"""
Runtime security checker: applies policy constraints to a given intent/user/resource.

Use `policy_planner.py` at startup to fetch wiki, build a plan, and extract policies.
Then feed the extracted constraints (from policies.json) into `classify` for each request.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal


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


def classify(intent: str, user_ctx: dict, resource_ctx: dict, policy: dict | Dict[str, Dict[str, Any]]) -> Decision:
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


__all__ = ["Decision", "classify", "load_policy_constraints"]
