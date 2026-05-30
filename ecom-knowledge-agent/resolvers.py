"""Phase 2 deterministic resolvers — "repair, not reject".

Path Resolver: when a grounding_ref is a bare id/SKU or a /-rooted path that was
never observed, the runtime itself looks up the real object path in SQL and
substitutes it. The agent cites the id it reliably knows; code resolves it.

Prod-invariant discipline (see plan §0):
- No hardcoded table/column names. The identity map is *discovered* by sampling
  each table for a column whose values are /-rooted (the object path), so it
  survives schema rotation.
- The resolved path is read from SQL output by format-agnostic regex over
  /-rooted tokens — never by assuming a CSV column layout.
- Purely additive: if discovery yields no usable map (e.g. non-CSV sample
  output), every function degrades to a no-op and the gate behaves as before.
- Resolution runs through rt.sql_raw / rt.exec, so any path it surfaces is
  auto-captured into rt.paths by the wrapper's evidence ledger — which is why a
  substituted ref then passes the finalization gate.

The one environment assumption left here is the SQL *dialect* for table
enumeration (`sqlite_schema`). It is isolated to `_list_tables` and fails soft.
"""
from __future__ import annotations

import re

from ecom_discovery import SessionDiscovery
from ecom_runtime import EcomRuntime

_PATH_RE = re.compile(r"/[^\s,;\"'\]\)\}#]+")


def _list_tables(rt: EcomRuntime) -> list[str]:
    rows = rt.sql("SELECT name FROM sqlite_schema WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    return [r["name"] for r in rows if r.get("name")]


def build_identity_map(rt: EcomRuntime) -> dict[str, dict]:
    """Sample each table to find its /-rooted path column + lookup columns.

    Returns {table: {"path_col": str, "cols": [other columns]}} for tables that
    expose a path column. Best-effort: any table we can't sample is skipped."""
    out: dict[str, dict] = {}
    try:
        tables = _list_tables(rt)
    except Exception:
        return out
    for t in tables:
        try:
            sample = rt.sql(f'SELECT * FROM "{t}" LIMIT 1')
        except Exception:
            continue
        if not sample:
            continue
        row = sample[0]
        path_col = next(
            (c for c, v in row.items() if isinstance(v, str) and v.startswith("/")),
            None,
        )
        if not path_col:
            continue
        out[t] = {"path_col": path_col, "cols": [c for c in row if c != path_col]}
    return out


def get_identity_map(rt: EcomRuntime, discovery: SessionDiscovery) -> dict[str, dict]:
    """Run-scoped memoized identity map (built once, reused across trials)."""
    if not discovery.identity_built:
        try:
            discovery.identity_columns = build_identity_map(rt)
        except Exception:
            discovery.identity_columns = {}
        discovery.identity_built = True
    return discovery.identity_columns


def _lookup_keys(ref: str) -> list[str]:
    """Candidate identifier strings to resolve a ref by.

    For a bare id this is the id itself; for a /-rooted path it's the trailing
    segment and its extension-stripped stem (the id is usually in the filename).
    Oversized / multiline refs (e.g. a pasted CSV blob) are skipped as whole-string
    keys but still mined for a path segment."""
    keys: list[str] = []
    r = ref.strip()
    if r and "\n" not in r and len(r) <= 128:
        keys.append(r)
    if "/" in r:
        seg = r.split("#", 1)[0].rstrip("/").split("/")[-1]
        if seg:
            keys.append(seg)
            stem = seg.rsplit(".", 1)[0]
            if stem and stem != seg:
                keys.append(stem)
    return list(dict.fromkeys(keys))


def resolve_ref(rt: EcomRuntime, idmap: dict[str, dict], ref: str):
    """Resolve a ref to a real object path via SQL id->path lookup.

    Returns one of:
      ("resolved", path)        exactly one path found
      ("ambiguous", [paths])    multiple distinct paths
      ("zero", None)            id not found anywhere
    """
    for key in _lookup_keys(ref):
        val = key.replace("'", "''")
        found: set[str] = set()
        for table, info in idmap.items():
            cols = info.get("cols") or []
            if not cols:
                continue
            clause = " OR ".join(f'"{c}" = \'{val}\'' for c in cols)
            query = f'SELECT "{info["path_col"]}" FROM "{table}" WHERE {clause} LIMIT 5;'
            try:
                out = rt.sql_raw(query)
            except Exception:
                continue
            found.update(_PATH_RE.findall(out))
        if len(found) == 1:
            return "resolved", next(iter(found))
        if len(found) > 1:
            return "ambiguous", sorted(found)
    return "zero", None


_AMBIGUOUS_EXPANSION_CAP = 20


def repair_grounding_refs(rt: EcomRuntime, discovery: SessionDiscovery, report) -> list[str]:
    """Substitute resolvable bare-id / unseen-path refs with real object paths.

    Mutates report.grounding_refs in place. Returns a list of "Synthetic
    Correction" log lines (code, not the LLM, supplied the final precision) so the
    trace stays honest.

    Cases:
      - ref already resolves to a known /-rooted path or a /docs path → kept.
      - bare id resolves to exactly one path → substituted (1-for-1).
      - bare id resolves to many paths (e.g. a family_id → its products) → the
        single ref is expanded into the list of paths, capped at
        _AMBIGUOUS_EXPANSION_CAP. This is the typical foreign-key fan-out and
        the gate wants concrete paths, so emit them all rather than leaving
        the bare ID in to be rejected.
      - bare id resolves to nothing → left untouched (the gate rejects it)."""
    idmap = get_identity_map(rt, discovery)
    if not idmap:
        return []

    known = rt.paths | set(discovery.docs_tree)
    corrections: list[str] = []
    new_refs: list[str] = []
    seen: set[str] = set()  # dedup expanded refs across multiple bare-id inputs

    def _append(p: str) -> None:
        if p not in seen:
            seen.add(p)
            new_refs.append(p)

    for ref in report.grounding_refs or []:
        base = ref.split("#", 1)[0].rstrip("/")
        if ref.startswith("/") and (base in known or base.startswith("/docs/")):
            _append(ref)
            continue
        status, payload = resolve_ref(rt, idmap, ref)
        if status == "resolved":
            _append(payload)
            corrections.append(f"Synthetic Correction: '{ref}' -> '{payload}' (SQL id->path)")
        elif status == "ambiguous":
            paths = payload[:_AMBIGUOUS_EXPANSION_CAP]
            for p in paths:
                _append(p)
            note = (
                f"Synthetic Correction: '{ref}' -> {len(paths)} paths "
                f"(SQL id->paths, fan-out)"
            )
            if len(payload) > _AMBIGUOUS_EXPANSION_CAP:
                note += f"; truncated from {len(payload)}"
            corrections.append(note)
        else:
            _append(ref)
    report.grounding_refs = new_refs
    return corrections
