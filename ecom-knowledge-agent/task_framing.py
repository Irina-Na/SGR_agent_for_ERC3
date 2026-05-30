"""Per-trial /docs and SQL-schema refresh.

The runtime policy/prompt selection itself lives in `policy_collector.py` — this
module only owns the invariant-3 refresh (`tree /docs` at trial start) and the
schema re-dump, both of which must be re-run per trial because doc filenames and
SQL shapes rotate per prod swap. Nothing in here may encode task-specific rules,
domain vocabulary, or doc names.
"""
from __future__ import annotations

from typing import List

from api_tools import Req_Tree
from bitgn.vm.ecom.ecom_pb2 import NodeKind
from ecom_discovery import SessionDiscovery


def _walk_docs(node, prefix: str, out: List[str]) -> None:
    for child in node.children:
        p = f"{prefix}/{child.name}"
        if child.kind == NodeKind.NODE_KIND_FILE:
            out.append(p)
        elif child.kind == NodeKind.NODE_KIND_DIR:
            _walk_docs(child, p, out)


def refresh_docs_for_trial(rt, discovery: SessionDiscovery) -> SessionDiscovery:
    """Per-trial `/docs` refresh (invariant 3: `tree /docs` at trial start).

    Returns a SessionDiscovery copy whose `docs_tree` reflects THIS trial's `/docs`
    (not the run-hoisted snapshot, which can drift). `policy_doc_index` is
    filtered to surviving paths so stale triggers can't misdirect downstream
    selection; new paths get an empty trigger list.

    Fail-soft: on any RPC/walk failure, returns the original discovery unchanged.
    """
    try:
        result = rt.tree(Req_Tree(tool="tree", root="/docs", level=0))
    except Exception:
        return discovery
    paths: List[str] = []
    try:
        _walk_docs(result.root, "/docs", paths)
    except Exception:
        return discovery
    if not paths:
        return discovery
    surviving_index = {
        p: discovery.policy_doc_index.get(p, [])
        for p in paths
    }
    return discovery.model_copy(update={
        "docs_tree": paths,
        "policy_doc_index": surviving_index,
    })


def refresh_schema_for_trial(rt, discovery: SessionDiscovery) -> SessionDiscovery:
    """Per-trial SQL schema refresh — schemas can rotate between trials in the
    same run (prime directive: table/column names are mutable). Re-dump via the
    runtime's sql_raw so the call counts against the evidence ledger like any
    other tool use.

    Returns a copy with the refreshed `schema_snapshot`; on any failure or empty
    result, returns the original discovery unchanged.
    """
    if not discovery.sql_tool:
        return discovery
    try:
        base = rt.sql_raw(
            "SELECT name, type, sql FROM sqlite_schema WHERE sql IS NOT NULL "
            "ORDER BY type, name;"
        )
    except Exception:
        return discovery
    if not base.strip():
        return discovery
    import re as _re
    table_re = _re.compile(
        r"create\s+table\s+(?:if\s+not\s+exists\s+)?[\"\[`]?([A-Za-z_][A-Za-z0-9_]*)",
        _re.IGNORECASE,
    )
    chunks = ["# sqlite_schema\n" + base.rstrip()]
    for name in sorted(set(table_re.findall(base))):
        try:
            cols = rt.sql_raw(f"PRAGMA table_info({name});")
        except Exception:
            continue
        if cols.strip():
            chunks.append(f"# PRAGMA table_info({name})\n{cols.rstrip()}")
    snapshot = "\n\n".join(chunks)
    return discovery.model_copy(update={"schema_snapshot": snapshot})


def format_schema_diag(baseline_len: int, trial_len: int) -> str:
    """One-line diag: baseline vs trial schema size + char delta for grep."""
    delta = trial_len - baseline_len
    return (
        f"SCHEMA DIAG: baseline_chars={baseline_len} trial_chars={trial_len} "
        f"delta={delta:+d}"
    )
