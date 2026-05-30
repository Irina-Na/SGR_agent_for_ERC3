"""
Session-scoped discovery. Runs once per BitGN run (not per trial).

Five eager RPCs against the live VM + one structured LLM classification call
that extracts a `SessionDiscovery` record from AGENTS.MD + the tree outputs.

No persistent cache, no offline harvest — the wiki layer is shared across
trials within a run, so in-memory reuse is sufficient.
"""
from __future__ import annotations

import json
import re
from typing import Dict, List, Literal, Optional

from openai import OpenAI
from pydantic import BaseModel, Field

from bitgn.vm.ecom.ecom_connect import EcomRuntimeClientSync
from bitgn.vm.ecom.ecom_pb2 import ExecRequest, NodeKind, Outcome, ReadRequest, TreeRequest


class ToolSpec(BaseModel):
    name: str
    path: str
    classification: Literal["mutating", "read_only", "unknown"] = "unknown"
    help_text: str = ""
    readme_path: Optional[str] = None


class SessionDiscovery(BaseModel):
    """Run-scoped knowledge about the ECOM environment."""

    tool_index: Dict[str, Literal["mutating", "read_only", "unknown"]] = Field(default_factory=dict)
    tool_specs: Dict[str, ToolSpec] = Field(default_factory=dict)
    identity_tool: Optional[str] = None
    time_tool: Optional[str] = None
    sql_tool: Optional[str] = None
    entity_kinds: List[str] = Field(default_factory=list)
    docs_tree: List[str] = Field(default_factory=list)
    policy_doc_index: Dict[str, List[str]] = Field(default_factory=dict)
    outcome_enum: List[str] = Field(default_factory=list)

    # raw artifacts kept for prompt construction
    agents_md: str = ""
    bin_tree_text: str = ""
    docs_tree_text: str = ""
    proc_tree_text: str = ""

    # Identity map for the Path Resolver, built lazily on first finalization-gate
    # use and memoized run-scoped: {table: {"path_col": str, "cols": [str, ...]}}.
    # Schema-agnostic — discovered by sampling, never hardcoded (see prime directive).
    identity_columns: Dict[str, dict] = Field(default_factory=dict)
    identity_built: bool = False

    # Run-scoped SQL schema dump (raw stdout from the discovered sql_tool). The
    # format is NOT a prod invariant — we ship the bytes and let the model read
    # them. Empty string when sql_tool wasn't discovered or the dump failed.
    schema_snapshot: str = ""


# ---- structured-output schema for the one LLM classification call ----
# Avoid Dict[str, Literal] in the response_format because OpenAI's structured
# outputs handle list-of-pairs more reliably than dict-with-additionalProperties.


class _ToolClass(BaseModel):
    tool_name: str
    classification: Literal["mutating", "read_only", "unknown"]


class _DocTriggers(BaseModel):
    doc_path: str
    triggers: List[str]


class _ClassificationResult(BaseModel):
    tool_classifications: List[_ToolClass]
    identity_tool: Optional[str]
    time_tool: Optional[str]
    sql_tool: Optional[str]
    doc_triggers: List[_DocTriggers]


# ---- helpers ----

def _format_tree_entry(entry, prefix: str = "", is_last: bool = True) -> List[str]:
    branch = "`-- " if is_last else "|-- "
    lines = [f"{prefix}{branch}{entry.name}"]
    child_prefix = f"{prefix}{'    ' if is_last else '|   '}"
    children = list(entry.children)
    for idx, child in enumerate(children):
        lines.extend(_format_tree_entry(child, child_prefix, idx == len(children) - 1))
    return lines


def _tree_text(vm: EcomRuntimeClientSync, root: str, level: int) -> str:
    result = vm.tree(TreeRequest(root=root, level=level))
    r = result.root
    if not r.name:
        return "."
    lines = [r.name]
    children = list(r.children)
    for idx, child in enumerate(children):
        lines.extend(_format_tree_entry(child, is_last=idx == len(children) - 1))
    return "\n".join(lines)


def _collect_proc_kinds(vm: EcomRuntimeClientSync) -> List[str]:
    result = vm.tree(TreeRequest(root="/proc", level=1))
    return [c.name for c in result.root.children if c.kind == NodeKind.NODE_KIND_DIR]


def _collect_docs_paths(vm: EcomRuntimeClientSync) -> List[str]:
    # Collect every FILE under /docs regardless of extension. Invariant 3 guarantees
    # /docs exists but not that policy docs are markdown — do not filter by extension.
    result = vm.tree(TreeRequest(root="/docs", level=0))
    paths: List[str] = []

    def walk(node, prefix: str) -> None:
        for child in node.children:
            p = f"{prefix}/{child.name}"
            if child.kind == NodeKind.NODE_KIND_FILE:
                paths.append(p)
            elif child.kind == NodeKind.NODE_KIND_DIR:
                walk(child, p)

    walk(result.root, "/docs")
    return paths


def _collect_bin_tools(vm: EcomRuntimeClientSync) -> List[str]:
    """Collect /bin leaf tool names from the live tree."""
    result = vm.tree(TreeRequest(root="/bin", level=1))
    return [
        child.name for child in result.root.children
        if child.kind in (NodeKind.NODE_KIND_FILE, NodeKind.NODE_KIND_DIR)
    ]


def _read_optional(vm: EcomRuntimeClientSync, path: str) -> str:
    try:
        return vm.read(ReadRequest(path=path)).content or ""
    except Exception:  # noqa: BLE001
        return ""


def _exec_optional(vm: EcomRuntimeClientSync, path: str, args: List[str]) -> str:
    try:
        result = vm.exec(ExecRequest(path=path, args=args, stdin=""))
    except Exception:  # noqa: BLE001
        return ""
    parts = []
    if getattr(result, "stdout", ""):
        parts.append(result.stdout.rstrip())
    if getattr(result, "stderr", ""):
        parts.append("stderr:\n" + result.stderr.rstrip())
    return "\n".join(parts).strip()


def _collect_tool_specs(
    vm: EcomRuntimeClientSync,
    tool_names: List[str],
    classifications: Dict[str, Literal["mutating", "read_only", "unknown"]],
) -> Dict[str, ToolSpec]:
    specs: Dict[str, ToolSpec] = {}
    for name in tool_names:
        path = f"/bin/{name}"
        readme_path = f"{path}/README.md"
        readme = _read_optional(vm, readme_path)
        help_text = readme
        if not help_text.strip():
            help_text = _exec_optional(vm, path, ["--help"])
        specs[name] = ToolSpec(
            name=name,
            path=path,
            classification=classifications.get(name, "unknown"),
            help_text=help_text[:4000],
            readme_path=readme_path if readme else None,
        )
    return specs


def _outcome_enum_names() -> List[str]:
    return [v.name for v in Outcome.DESCRIPTOR.values]


_TABLE_NAME_RE = re.compile(
    r"create\s+table\s+(?:if\s+not\s+exists\s+)?[\"\[`]?([A-Za-z_][A-Za-z0-9_]*)",
    re.IGNORECASE,
)


def _sql_exec(vm: EcomRuntimeClientSync, sql_tool: str, query: str) -> str:
    """Run a query via the discovered SQL tool; return raw stdout, "" on error."""
    try:
        r = vm.exec(ExecRequest(path=f"/bin/{sql_tool}", args=[], stdin=query))
        return getattr(r, "stdout", "") or ""
    except Exception:  # noqa: BLE001
        return ""


_INTROSPECTION_QUERIES = (
    # SQLite
    ("sqlite_schema",
     "SELECT name, type, sql FROM sqlite_schema WHERE sql IS NOT NULL "
     "ORDER BY type, name;"),
    # Postgres / MySQL / standard SQL
    ("information_schema.tables",
     "SELECT table_schema, table_name FROM information_schema.tables "
     "WHERE table_schema NOT IN "
     "('pg_catalog','information_schema','sys','mysql','performance_schema') "
     "ORDER BY table_schema, table_name;"),
)


def dump_schema(sql_exec_fn, entity_kinds=None) -> str:
    """Raw schema dump for the system prompt. Format-agnostic, multi-dialect.

    `sql_exec_fn(query)` -> raw stdout str (""/None on failure; must not raise).

    Strategy: try standard introspection queries (SQLite + information_schema),
    then PRAGMA-probe any names parsed out, then always probe `SELECT * LIMIT 1`
    against every discovered entity_kind (and its dash/underscore variants).
    The entity-kind probes are the prod-safe baseline — they rely only on
    invariant 4 (objects in /proc are mirrored in SQL) and work on any backend
    that supports `SELECT * FROM <table> LIMIT 1`, surfacing column headers
    even when no metadata view is queryable.

    The result is one concatenated string; the model reads it as text. Empty
    only when every probe returned nothing.
    """
    chunks: List[str] = []
    detected: List[str] = []

    for label, query in _INTROSPECTION_QUERIES:
        out = sql_exec_fn(query) or ""
        if not out.strip():
            continue
        chunks.append(f"# {label}\n{out.rstrip()}")
        if label == "sqlite_schema":
            detected.extend(_TABLE_NAME_RE.findall(out))
        else:
            # best-effort: last comma-separated token per line is table_name
            for line in out.splitlines()[1:]:
                parts = [p.strip().strip('"').strip("'") for p in line.split(",")]
                if len(parts) >= 2 and parts[-1]:
                    detected.append(parts[-1])

    for name in sorted({n for n in detected if n}):
        cols = sql_exec_fn(f"PRAGMA table_info({name});") or ""
        if cols.strip():
            chunks.append(f"# PRAGMA table_info({name})\n{cols.rstrip()}")

    seen = {n for n in detected if n}
    for ek in (entity_kinds or []):
        for variant in (ek, ek.replace("-", "_"), ek.replace("_", "-")):
            if not variant or variant in seen:
                continue
            seen.add(variant)
            sample = sql_exec_fn(f"SELECT * FROM {variant} LIMIT 1;") or ""
            if sample.strip():
                chunks.append(f"# entity-kind probe: {variant}\n{sample.rstrip()}")

    return "\n\n".join(chunks)


def _dump_schema(vm: EcomRuntimeClientSync, sql_tool: Optional[str],
                 entity_kinds: Optional[List[str]] = None) -> str:
    """discover()-time wrapper around `dump_schema` using the raw VM client."""
    if not sql_tool:
        return ""
    return dump_schema(
        lambda q: _sql_exec(vm, sql_tool, q),
        entity_kinds=entity_kinds,
    )


def _extract_json_candidate(content: str) -> str:
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
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and start < end:
        return text[start: end + 1].strip()
    return text


_SYSTEM_PROMPT = """You are classifying an ECOM runtime environment. You are given:
- The root /AGENTS.MD content (describes tools, conventions, key locations)
- tree output for /bin (utility inventory)
- tree output for /docs (policy doc list)
- tree output for /proc (entity kinds)

Produce a strict structured classification:

- tool_classifications: for each entry in /bin (use the leaf name, NOT the full path), classify as:
    "mutating"   -> changes state (apply / approve / write / finalize / checkout / refund / delete)
    "read_only"  -> only reads (date / id / whoami / sql query / status / list)
    "unknown"    -> AGENTS.MD does not describe its role
- identity_tool: the /bin leaf name that reports the current user / session identity, or null
- time_tool: the /bin leaf name that reports current date / time, or null
- sql_tool: the /bin leaf name that accepts SQL on stdin, or null
- doc_triggers: for each file under /docs (any type), a short list of domain trigger
  keywords inferred from the filename and path (e.g. "fraud", "refund", "discount",
  "3DS", "return", "checkout", "incident", "addendum", "security"). Empty list is acceptable.

Use AGENTS.MD's natural-language descriptions as the primary source of truth.
"""


def discover(
    vm: EcomRuntimeClientSync,
    llm_client: OpenAI,
    model: str,
) -> SessionDiscovery:
    """Run the discovery sequence and return a populated SessionDiscovery."""
    # 1-5: five eager RPCs
    _root_tree = _tree_text(vm, "/", 2)  # noqa: F841 — informational, not used downstream
    agents_md = vm.read(ReadRequest(path="/AGENTS.MD")).content
    bin_tree = _tree_text(vm, "/bin", 2)
    docs_tree = _tree_text(vm, "/docs", 2)
    proc_tree = _tree_text(vm, "/proc", 2)

    entity_kinds = _collect_proc_kinds(vm)
    docs_paths = _collect_docs_paths(vm)
    bin_tools = _collect_bin_tools(vm)

    # 6: one LLM call — JSON-schema-in-prompt + manual validate (same style as the
    # main loop; avoids constrained decoding that the Nebius/Qwen model handles poorly)
    schema = json.dumps(_ClassificationResult.model_json_schema(), ensure_ascii=False)
    system = (
        f"{_SYSTEM_PROMPT.rstrip()}\n\n"
        "Return only one valid JSON object. No markdown, no prose, no code fences.\n"
        f"The JSON must validate against this schema:\n{schema}"
    )
    user_msg = (
        f"# /AGENTS.MD\n{agents_md}\n\n"
        f"# tree /bin\n{bin_tree}\n\n"
        f"# tree /docs\n{docs_tree}\n\n"
        f"# tree /proc\n{proc_tree}\n"
    )
    resp = llm_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ],
        max_completion_tokens=16384,
    )
    content = resp.choices[0].message.content or ""
    try:
        cls = _ClassificationResult.model_validate_json(_extract_json_candidate(content))
    except Exception as exc:  # noqa: BLE001
        # safety net — keep going with empty taxonomies; the agent still functions,
        # just without classification metadata
        print(f"discovery classification failed ({exc}); continuing with empty taxonomies")
        cls = _ClassificationResult(
            tool_classifications=[],
            identity_tool=None,
            time_tool=None,
            sql_tool=None,
            doc_triggers=[],
        )

    tool_index = {tc.tool_name: tc.classification for tc in cls.tool_classifications}
    # LLM classification may omit tools; keep every /bin leaf visible as unknown.
    for name in bin_tools:
        tool_index.setdefault(name, "unknown")
    tool_specs = _collect_tool_specs(vm, bin_tools, tool_index)

    schema_snapshot = _dump_schema(vm, cls.sql_tool, entity_kinds=entity_kinds)

    return SessionDiscovery(
        tool_index=tool_index,
        tool_specs=tool_specs,
        identity_tool=cls.identity_tool,
        time_tool=cls.time_tool,
        sql_tool=cls.sql_tool,
        entity_kinds=entity_kinds,
        docs_tree=docs_paths,
        policy_doc_index={dt.doc_path: dt.triggers for dt in cls.doc_triggers},
        outcome_enum=_outcome_enum_names(),
        agents_md=agents_md,
        bin_tree_text=bin_tree,
        docs_tree_text=docs_tree,
        proc_tree_text=proc_tree,
        schema_snapshot=schema_snapshot,
    )
