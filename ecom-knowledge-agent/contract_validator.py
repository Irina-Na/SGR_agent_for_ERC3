"""
Deterministic contract enforcement: evidence ledger + finalization gate.

The ledger records every /-rooted object/doc path that actually appeared in a
tool result this trial. The gate intercepts `report_completion` and rejects:

  1. ref-validity   - a grounding_ref that was never observed (hallucinated path)
                      or is table/column shorthand instead of a /-rooted path
  2. docs-citation  - a policy was applied (a /docs/* doc was read, or the message
                      reads as a policy/refusal) but no /docs ref is cited
  3. outcome        - the message reads as a refusal/denial but outcome == OUTCOME_OK

All checks are discovery-driven; nothing hardcodes a specific doc filename, table,
or role token (production may rotate them).
"""
from __future__ import annotations

import re
from typing import Iterable

from api_tools import ReportTaskCompletion
from ecom_discovery import SessionDiscovery


# /-rooted path ending in .json or .md, stopping at whitespace / CSV / quote / bracket delims
_PATH_RE = re.compile(r"/[^\s,;\"'\]\)\}]+\.(?:json|md)")

# table.column shorthand the grader rejects (e.g. "products.path", "inventory")
_SHORTHAND_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")

# refusal / denial language (generalized; not tied to a specific policy filename)
_REFUSAL_TERMS = (
    "cannot disclose", "can't disclose", "cannot provide", "can't provide",
    "cannot share", "will not share", "not authorized", "not allowed",
    "not permitted", "cannot be disclosed", "decline", "denied", "deny",
    "refuse", "per our security", "security policy", "against policy",
    "violates policy", "for security reasons",
)


def harvest_paths(*texts: str) -> set[str]:
    """Extract every /-rooted .json/.md path token from one or more text blobs."""
    found: set[str] = set()
    for t in texts:
        if not t:
            continue
        found.update(_PATH_RE.findall(t))
    return found


def _looks_like_refusal(message: str) -> bool:
    low = message.lower()
    return any(term in low for term in _REFUSAL_TERMS)


def _is_docs_path(path: str, docs_tree: Iterable[str]) -> bool:
    if path.startswith("/docs/"):
        return True
    return path in set(docs_tree)


def validate_report(
    report: ReportTaskCompletion,
    seen_paths: set[str],
    discovery: SessionDiscovery,
    docs_read_this_trial: set[str],
) -> list[str]:
    """Return a list of violation strings. Empty list == passes the gate."""
    violations: list[str] = []
    refs = list(report.grounding_refs or [])

    # paths confirmed to exist: those seen in tool results this trial, plus the
    # /docs tree enumerated during discovery (tree-confirmed real files).
    known_valid = seen_paths | set(discovery.docs_tree)

    # 1. ref-validity
    for ref in refs:
        if not ref.startswith("/"):
            kind = "table/column shorthand" if _SHORTHAND_RE.match(ref) else "not a /-rooted path"
            violations.append(
                f"grounding_ref '{ref}' is {kind}. Cite the concrete /-rooted path "
                f"from a tool result (e.g. the `path` column value)."
            )
        elif ref not in known_valid:
            violations.append(
                f"grounding_ref '{ref}' was never returned by any tool this trial "
                f"(possible hallucinated path). Only cite paths that appeared in a read/SQL/tree result."
            )

    # 2. docs-citation: if a policy was applied, require a /docs ref
    applied_policy = bool(docs_read_this_trial) or _looks_like_refusal(report.message)
    if applied_policy:
        has_docs_ref = any(_is_docs_path(r, discovery.docs_tree) for r in refs)
        if not has_docs_ref:
            hint = ""
            if docs_read_this_trial:
                sample = sorted(docs_read_this_trial)[:3]
                hint = f" You read: {', '.join(sample)} — cite the one you applied."
            violations.append(
                "the answer applies/relies on a policy but cites no /docs document." + hint
            )

    # 3. outcome consistency (advisory: refusal must not be OUTCOME_OK)
    if _looks_like_refusal(report.message) and report.outcome == "OUTCOME_OK":
        violations.append(
            "the message reads as a refusal/denial but outcome=OUTCOME_OK. "
            "If you are refusing on policy/security grounds, use OUTCOME_DENIED_SECURITY; "
            "if you need more info from the user, use OUTCOME_NONE_CLARIFICATION."
        )

    return violations
