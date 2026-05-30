"""Typed execution layer over the ECOM VM client (Phase 1).

`EcomRuntime` is the single I/O surface for everything the agent does. Execution
lives behind typed methods instead of string-shaped requests, and the evidence
ledger (citeable file paths, /docs pages read) is populated INSIDE the wrappers
on every call — so it survives output-format changes and is captured identically
whether the caller is the step-loop dispatch or (Schema B) a sandboxed script.

The same instance is the object a script would receive: rt.read(...), rt.sql(...),
rt.list(...), etc. Raw RPC results are returned unchanged so the presentation
layer in agent.py keeps working as-is; `sql()` is the one convenience wrapper that
parses rows to list[dict] for resolvers / scripts.
"""
from __future__ import annotations

import csv
import io
import json
import re

from google.protobuf.json_format import MessageToDict

from bitgn.vm.ecom.ecom_connect import EcomRuntimeClientSync
from bitgn.vm.ecom.ecom_pb2 import (
    AnswerRequest,
    DeleteRequest,
    ExecRequest,
    FindRequest,
    ListRequest,
    NodeKind,
    ReadRequest,
    SearchRequest,
    StatRequest,
    TreeRequest,
    WriteRequest,
)

from api_tools import (
    OUTCOME_BY_NAME,
    Req_Delete,
    Req_Exec,
    Req_Find,
    Req_List,
    Req_Read,
    Req_Search,
    Req_Stat,
    Req_Tree,
    Req_Write,
    ReportTaskCompletion,
)

# /-rooted token for exec/SQL stdout. No extension requirement: SQL `path` cells are
# citeable regardless of extension. Over-capture here only relaxes the gate (avoids
# false rejection); bare-dir protection stays on the structured FILE-kind tools below.
_EXEC_PATH_RE = re.compile(r"/[^\s,;\"'\]\)\}#]+")

_KIND_MAP = {
    "all": NodeKind.NODE_KIND_UNSPECIFIED,
    "files": NodeKind.NODE_KIND_FILE,
    "dirs": NodeKind.NODE_KIND_DIR,
}


def _join(base: str, name: str) -> str:
    return (base.rstrip("/") + "/" + name) if base not in ("", "/") else "/" + name


def _walk_tree_files(node, cur_path: str, out: set[str]) -> None:
    for child in node.children:
        child_path = _join(cur_path, child.name)
        if child.kind == NodeKind.NODE_KIND_FILE:
            out.add(child_path)
        elif child.kind == NodeKind.NODE_KIND_DIR:
            _walk_tree_files(child, child_path, out)


def harvest_from_result(cmd, result) -> set[str]:
    """Extract citeable FILE paths from a structured RPC result.

    Uses the runtime's own NodeKind (FILE vs DIR) where available, so it survives
    a change in object-path extensions in prod. Directories and tool paths are
    never added (they aren't FILE nodes and don't appear as SQL path cells)."""
    out: set[str] = set()
    if result is None:
        return out
    if isinstance(cmd, Req_Read):
        out.add(cmd.path)
    elif isinstance(cmd, Req_Write):
        out.add(cmd.path)
    elif isinstance(cmd, Req_Tree):
        _walk_tree_files(result.root, cmd.root or "/", out)
    elif isinstance(cmd, Req_List):
        for e in result.entries:
            if e.kind == NodeKind.NODE_KIND_FILE:
                out.add(_join(cmd.path or "/", e.name))
    elif isinstance(cmd, Req_Search):
        for m in result.matches:
            out.add(m.path)
    elif isinstance(cmd, Req_Exec):
        out.update(_EXEC_PATH_RE.findall(getattr(result, "stdout", "") or ""))
    elif isinstance(cmd, Req_Find):
        try:
            out.update(_EXEC_PATH_RE.findall(json.dumps(MessageToDict(result))))
        except Exception:
            pass
    return out


class EcomRuntime:
    """Typed I/O surface + evidence ledger over the VM client."""

    def __init__(
        self,
        vm: EcomRuntimeClientSync,
        sql_tool: str | None = None,
        tool_specs: dict | None = None,
    ):
        self.vm = vm
        # Prod invariant: utilities live in /bin, but their NAMES are discovered,
        # not fixed. Never hardcode a tool name — if discovery didn't find the SQL
        # tool, sql() refuses rather than guessing a path that may not exist.
        self.sql_tool_path = f"/bin/{sql_tool}" if sql_tool else None
        self.tool_specs = tool_specs or {}
        self.paths: set[str] = set()      # citeable file paths seen this trial
        self.docs_read: set[str] = set()  # /docs pages actually read

    # ---- ledger ----
    def _record(self, cmd, result, *, count_as_policy: bool = True) -> None:
        self.paths.update(harvest_from_result(cmd, result))
        if count_as_policy and isinstance(cmd, Req_Read) and cmd.path.startswith("/docs/"):
            self.docs_read.add(cmd.path)

    # ---- typed wrappers (each records evidence, returns the raw RPC result) ----
    def tree(self, cmd: Req_Tree):
        r = self.vm.tree(TreeRequest(root=cmd.root, level=cmd.level))
        self._record(cmd, r)
        return r

    def find(self, cmd: Req_Find):
        r = self.vm.find(FindRequest(
            root=cmd.root, name=cmd.name, kind=_KIND_MAP[cmd.kind], limit=cmd.limit,
        ))
        self._record(cmd, r)
        return r

    def search(self, cmd: Req_Search):
        r = self.vm.search(SearchRequest(root=cmd.root, pattern=cmd.pattern, limit=cmd.limit))
        self._record(cmd, r)
        return r

    def list(self, cmd: Req_List):
        r = self.vm.list(ListRequest(path=cmd.path))
        self._record(cmd, r)
        return r

    def read(self, cmd: Req_Read):
        r = self.vm.read(ReadRequest(
            path=cmd.path, number=cmd.number,
            start_line=cmd.start_line, end_line=cmd.end_line,
        ))
        self._record(cmd, r)
        return r

    def read_for_context(self, cmd: Req_Read):
        """Read used as background context (e.g. policy-collector pre-read).

        Updates `paths` so the result is still citeable, but does NOT mark the
        doc as "policy applied" — only an agent-executed script doing rt.read
        should set off the docs-citation gate. Without this split, the collector
        reading top-K candidates would force a docs-cite violation on every task.
        """
        r = self.vm.read(ReadRequest(
            path=cmd.path, number=cmd.number,
            start_line=cmd.start_line, end_line=cmd.end_line,
        ))
        self._record(cmd, r, count_as_policy=False)
        return r

    def write(self, cmd: Req_Write):
        r = self.vm.write(WriteRequest(path=cmd.path, content=cmd.content))
        self._record(cmd, r)
        return r

    def delete(self, cmd: Req_Delete):
        r = self.vm.delete(DeleteRequest(path=cmd.path))
        self._record(cmd, r)
        return r

    def stat(self, cmd: Req_Stat):
        r = self.vm.stat(StatRequest(path=cmd.path))
        self._record(cmd, r)
        return r

    def exec(self, cmd: Req_Exec):
        r = self.vm.exec(ExecRequest(path=cmd.path, args=cmd.args, stdin=cmd.stdin))
        self._record(cmd, r)
        return r

    def tool_help(self, name: str) -> str:
        spec = self.tool_specs.get(name) or self.tool_specs.get(name.strip("/").split("/")[-1])
        if spec is None:
            return ""
        if isinstance(spec, dict):
            return spec.get("help_text", "") or ""
        return getattr(spec, "help_text", "") or ""

    def run_tool(self, name: str, args=(), stdin: str = ""):
        leaf = name.strip("/").split("/")[-1]
        return self.exec(Req_Exec(
            tool="exec",
            path=f"/bin/{leaf}",
            args=list(args),
            stdin=stdin,
        ))

    def answer(self, cmd: ReportTaskCompletion):
        return self.vm.answer(AnswerRequest(
            message=cmd.message,
            outcome=OUTCOME_BY_NAME[cmd.outcome],
            refs=cmd.grounding_refs,
        ))

    # ---- router (replaces the old module-level dispatch) ----
    def execute(self, cmd):
        handler = {
            Req_Tree: self.tree, Req_Find: self.find, Req_Search: self.search,
            Req_List: self.list, Req_Read: self.read, Req_Write: self.write,
            Req_Delete: self.delete, Req_Stat: self.stat, Req_Exec: self.exec,
            ReportTaskCompletion: self.answer,
        }.get(type(cmd))
        if handler is None:
            raise ValueError(f"Unknown command: {cmd}")
        return handler(cmd)

    # ---- convenience for resolvers / scripts (Phase 2/3) ----
    def sql_raw(self, query: str) -> str:
        """Run a query through the discovered SQL tool; return raw stdout.

        This is the prod-robust primitive: it makes no assumption about output
        FORMAT (not one of the environment invariants). Path resolution should
        lean on the evidence ledger (rt.paths), which captures /-rooted cells
        format-agnostically via regex — both rely only on invariant (4) (objects
        in /proc, mirrored in SQL), never on a column layout."""
        if self.sql_tool_path is None:
            raise RuntimeError(
                "No SQL tool discovered this run; cannot run sql(). "
                "Discovery must supply sql_tool (a /bin leaf name)."
            )
        r = self.exec(Req_Exec(tool="exec", path=self.sql_tool_path, args=[], stdin=query))
        return getattr(r, "stdout", "") or ""

    def sql(self, query: str) -> list[dict]:
        """Best-effort structured rows from the SQL tool, as list[dict].

        Assumes CSV-with-header output, which is what the current env emits but is
        NOT a guaranteed prod invariant — so this degrades to [] on any other
        shape rather than raising. Use sql_raw() / rt.paths when you only need the
        /-rooted path cell (the format-agnostic, prod-safe route)."""
        stdout = self.sql_raw(query)
        if not stdout.strip():
            return []
        try:
            return list(csv.DictReader(io.StringIO(stdout)))
        except Exception:
            return []
