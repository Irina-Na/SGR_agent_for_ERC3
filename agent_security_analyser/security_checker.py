"""
Runtime security checker: applies policy constraints to a given intent/user/resource.

Use `policy_planner.py` at startup to fetch wiki, build a plan, and extract policies.
Then feed a normalized policy dict into `classify` for each request.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SENSITIVE_INTENTS = {"salary_view", "salary_change", "pii_view"}
PROJECT_MUTATIONS = {"project_status_change", "project_team_change"}
TIME_LOGGING = {"time_entry_create"}


@dataclass
class Decision:
    status: Literal["allow", "deny", "clarify"]
    reason: str


def classify(intent: str, user_ctx: dict, resource_ctx: dict, policy: dict) -> Decision:
    """
    Apply fail-closed classification of a request using a normalized policy dict.

    Expected policy format:
      {
        "constraints": {
          "<intent>": {
            "roles": [...],
            "override_roles": [...],
            "locations": [...],  # ["any"] to skip location check
            "needs_clarification": bool,
            "allow_if_exec": bool,
            "needs_responsibility": bool
          },
          ...
        },
        "defaults": {"fail_closed": True}
      }
    """
    constraints_map = policy.get("constraints", {}) if isinstance(policy, dict) else {}
    constraints = constraints_map.get(intent)
    if not constraints:
        return Decision("deny", f"policy_no_rule_for_{intent}")

    role = user_ctx.get("role")
    allowed_roles = constraints.get("roles", [])
    if allowed_roles and role not in allowed_roles:
        return Decision("deny", f"role_not_allowed:{role}")

    allowed_locations = constraints.get("locations")
    if allowed_locations and "any" not in allowed_locations:
        locs = {resource_ctx.get("project_location"), user_ctx.get("location")}
        if not any(loc in allowed_locations for loc in locs if loc):
            return Decision("deny", "location_scope_block")

    if intent in PROJECT_MUTATIONS or constraints.get("needs_responsibility"):
        if not resource_ctx.get("is_owner_or_lead", False) and role not in constraints.get(
            "override_roles", []
        ):
            return Decision("deny", "not_responsible_for_project")

    if intent in TIME_LOGGING:
        if not resource_ctx.get("user_on_project", False):
            return Decision("deny", "not_assigned_to_project")

    if intent in SENSITIVE_INTENTS:
        if not constraints.get("allow_if_exec", False) and role != "exec":
            return Decision("deny", "sensitive_data_blocked")

    if constraints.get("needs_clarification") and not resource_ctx.get("target_resolved"):
        return Decision("clarify", "need_disambiguation")

    return Decision("allow", "ok")


__all__ = ["Decision", "classify"]
