"""
Session-scoped discovery. Runs once per BitGN run (not per trial).

Five eager RPCs against the live VM + one structured LLM classification call
that extracts a `SessionDiscovery` record from AGENTS.MD + the tree outputs.

No persistent cache, no offline harvest — the wiki layer is shared across
trials within a run, so in-memory reuse is sufficient.
"""
from __future__ import annotations

from typing import Dict, List, Literal, Optional

from openai import OpenAI
from pydantic import BaseModel, Field

from bitgn.vm.ecom.ecom_connect import EcomRuntimeClientSync
from bitgn.vm.ecom.ecom_pb2 import NodeKind, Outcome, ReadRequest, TreeRequest


class SessionDiscovery(BaseModel):
    """Run-scoped knowledge about the ECOM environment."""

    tool_index: Dict[str, Literal["mutating", "read_only", "unknown"]] = Field(default_factory=dict)
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

    # populated lazily on first SQL call (not by discover())
    identity_columns: Dict[str, List[str]] = Field(default_factory=dict)


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
    result = vm.tree(TreeRequest(root="/docs", level=0))
    paths: List[str] = []

    def walk(node, prefix: str) -> None:
        for child in node.children:
            p = f"{prefix}/{child.name}"
            if child.kind == NodeKind.NODE_KIND_FILE and child.name.endswith(".md"):
                paths.append(p)
            elif child.kind == NodeKind.NODE_KIND_DIR:
                walk(child, p)

    walk(result.root, "/docs")
    return paths


def _outcome_enum_names() -> List[str]:
    return [v.name for v in Outcome.DESCRIPTOR.values]


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
- doc_triggers: for each /docs/*.md path, a short list of domain trigger keywords
  inferred from the filename and path (e.g. "fraud", "refund", "discount", "3DS",
  "return", "checkout", "incident", "addendum", "security"). Empty list is acceptable.

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

    # 6: one structured LLM call
    user_msg = (
        f"# /AGENTS.MD\n{agents_md}\n\n"
        f"# tree /bin\n{bin_tree}\n\n"
        f"# tree /docs\n{docs_tree}\n\n"
        f"# tree /proc\n{proc_tree}\n"
    )
    resp = llm_client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        response_format=_ClassificationResult,
    )
    cls = resp.choices[0].message.parsed
    if cls is None:
        # safety net — keep going with empty taxonomies; the agent will still function,
        # just without classification metadata
        cls = _ClassificationResult(
            tool_classifications=[],
            identity_tool=None,
            time_tool=None,
            sql_tool=None,
            doc_triggers=[],
        )

    return SessionDiscovery(
        tool_index={tc.tool_name: tc.classification for tc in cls.tool_classifications},
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
    )
