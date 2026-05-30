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
    same run (prime directive: table/column names are mutable).

    Uses the same multi-dialect dump as discovery so any backend that responds
    to `SELECT * FROM <entity_kind> LIMIT 1` produces a usable schema snapshot
    even when no metadata view is queryable. Returns the original discovery
    unchanged on empty/failure.
    """
    if not discovery.sql_tool:
        return discovery
    from ecom_discovery import dump_schema

    def _safe(q: str) -> str:
        try:
            return rt.sql_raw(q) or ""
        except Exception:
            return ""

    snapshot = dump_schema(_safe, entity_kinds=discovery.entity_kinds)
    if not snapshot:
        return discovery
    return discovery.model_copy(update={"schema_snapshot": snapshot})


def format_schema_diag(baseline_len: int, trial_len: int) -> str:
    """One-line diag: baseline vs trial schema size + char delta for grep."""
    delta = trial_len - baseline_len
    return (
        f"SCHEMA DIAG: baseline_chars={baseline_len} trial_chars={trial_len} "
        f"delta={delta:+d}"
    )
