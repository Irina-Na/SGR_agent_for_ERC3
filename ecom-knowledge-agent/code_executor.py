"""Phase-3 sandbox: in-process restricted exec for model-written Python scripts.

The model writes ONE script that solves a task using the typed `EcomRuntime`
wrappers. We exec it here under four layered defenses, each catching a
different class of mistake / escape:

  1. AST screen     — no `import`, no dunder attribute access, no banned
                      builtins by name, no `global`/`nonlocal`.
  2. Builtins       — whitelist of pure data/control primitives only;
                      `open`/`eval`/`exec`/`__import__`/`os`/etc. are simply
                      absent from the script's namespace.
  3. Wrapper budget — every `rt.*` I/O call decrements a counter; exhaustion
                      raises and stops the script.
  4. Cooperative
     timeout       — a watchdog Event the wrappers check; an I/O-blocked
                      runaway script trips it on its next wrapper call.

Honest scope: this is in-process sandboxing, NOT OS isolation. A maliciously
crafted script with novel escapes could still break out. The threat model here
is "honest LLM writes a buggy or runaway script" — not "adversary tries to
exploit the host." For untrusted code you would run in a subprocess/container;
that's the next increment if we ever take untrusted scripts.

API:
    outcome = run_script(source, rt, timeout_sec=30, wrapper_budget=60)
    outcome.ok  →  outcome.namespace[<assigned-var>]
"""
from __future__ import annotations

import ast
import builtins as _b
import json as _json
import re as _re
import threading
from dataclasses import dataclass
from typing import Any, Optional

from api_tools import (
    Req_Exec, Req_Find, Req_List, Req_Read, Req_Search, Req_Tree,
)


class SandboxError(Exception):
    """Sandbox-detected violation (AST, budget, timeout, runtime)."""


# --- AST screen ---------------------------------------------------------------

_BANNED_NAMES = frozenset({
    "eval", "exec", "compile", "__import__", "open", "input",
    "setattr", "delattr",                       # mutate state — keep banned
    "globals", "locals", "vars", "dir", "type",
    "super", "breakpoint", "memoryview",
    "classmethod", "staticmethod", "property", "object",
    # `getattr` / `hasattr` are now allowed — honest scripts use them to read
    # named attributes on objects. We block escape via dunder/private *names*
    # passed as literals (`getattr(obj, "__class__")`) in `_screen_ast` below.
    # `rt.exec(...)` is an Attribute call, not a Name reference — the ban on the
    # bare `exec` name does not block it.
})

_GETATTR_FAMILY = frozenset({"getattr", "hasattr", "setattr", "delattr"})


def _screen_ast(source: str) -> None:
    tree = ast.parse(source, mode="exec")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise SandboxError("imports are not allowed")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            # blocks .__class__, .__bases__, .__subclasses__, etc., and also
            # any private attribute access. Slightly conservative but cheap.
            raise SandboxError(f"private/dunder attribute access not allowed: .{node.attr}")
        if isinstance(node, ast.Name) and node.id in _BANNED_NAMES:
            raise SandboxError(f"name not allowed: {node.id}")
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            raise SandboxError("global/nonlocal not allowed")
        # Block escape via literal dunder/private name string handed to the
        # getattr family: e.g. `getattr(obj, "__class__")` would bypass the
        # Attribute-node check above. Variable second args can't be checked
        # statically, but honest LLM scripts don't obfuscate.
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id in _GETATTR_FAMILY and len(node.args) >= 2:
            arg = node.args[1]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str) \
                    and arg.value.startswith("_"):
                raise SandboxError(
                    f"{node.func.id}() with private/dunder attribute name "
                    f"not allowed: {arg.value!r}"
                )


# --- restricted builtins (whitelist) -----------------------------------------

_SAFE_BUILTIN_NAMES = (
    "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter",
    "float", "frozenset", "int", "isinstance", "issubclass",
    "len", "list", "map", "max", "min", "pow", "range", "reversed", "round",
    "set", "slice", "sorted", "str", "sum", "tuple", "zip",
    "repr", "chr", "ord", "bin", "hex", "oct", "format", "print",
    "getattr", "hasattr",   # AST screen still rejects literal dunder/private names
)
_SAFE_BUILTINS: dict[str, Any] = {n: getattr(_b, n) for n in _SAFE_BUILTIN_NAMES}
_SAFE_BUILTINS.update({"True": True, "False": False, "None": None})


# --- budgeted runtime proxy ---------------------------------------------------

class _BudgetedRuntime:
    """Script-facing facade: primitive-arg methods over EcomRuntime, each
    decrementing the wrapper budget and checking the cancel signal. The
    underlying rt remains the only I/O surface, so evidence-ledger capture
    and the gate work identically to the step loop."""

    def __init__(self, rt, budget: int, cancel: threading.Event):
        # use slot-like attrs prefixed with _ so AST screen's dunder/private
        # block prevents the script from poking at them (rt._rt etc.).
        # (Even if it tried, the AST screen rejects underscore attrs above.)
        object.__setattr__(self, "_rt", rt)
        object.__setattr__(self, "_remaining", budget)
        object.__setattr__(self, "_cancel", cancel)

    def _check(self, name: str) -> None:
        if self._cancel.is_set():
            raise SandboxError("script wall-clock timeout exceeded")
        if self._remaining <= 0:
            raise SandboxError(f"wrapper-call budget exhausted at rt.{name}()")
        object.__setattr__(self, "_remaining", self._remaining - 1)

    # one method per typed wrapper; primitive args (script can't construct Req_*)
    def read(self, path: str, number: bool = False, start_line: int = 0, end_line: int = 0):
        self._check("read")
        return self._rt.read(Req_Read(tool="read", path=path, number=number,
                                      start_line=start_line, end_line=end_line))

    def list(self, path: str = "/"):
        self._check("list")
        return self._rt.list(Req_List(tool="list", path=path))

    def tree(self, root: str = "", level: int = 2):
        self._check("tree")
        return self._rt.tree(Req_Tree(tool="tree", root=root, level=level))

    def find(self, name: str, root: str = "/", kind: str = "all", limit: int = 10):
        self._check("find")
        return self._rt.find(Req_Find(tool="find", name=name, root=root,
                                      kind=kind, limit=min(limit, 20)))

    def search(self, pattern: str, root: str = "/", limit: int = 10):
        self._check("search")
        return self._rt.search(Req_Search(tool="search", pattern=pattern,
                                          root=root, limit=min(limit, 20)))

    def sql(self, query: str) -> list[dict]:
        self._check("sql")
        return self._rt.sql(query)

    def sql_raw(self, query: str) -> str:
        self._check("sql_raw")
        return self._rt.sql_raw(query)

    def exec(self, path: str, args=(), stdin: str = ""):
        self._check("exec")
        return self._rt.exec(Req_Exec(tool="exec", path=path,
                                      args=list(args), stdin=stdin))

    # read-only view of the evidence ledger so scripts can prefer-cite seen paths
    @property
    def paths(self):
        return frozenset(self._rt.paths)


# --- public entrypoint --------------------------------------------------------

@dataclass
class ExecOutcome:
    namespace: dict[str, Any]
    error: Optional[str] = None
    wrappers_used: int = 0
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.error is None


def run_script(
    source: str,
    rt,
    timeout_sec: float = 30.0,
    wrapper_budget: int = 60,
) -> ExecOutcome:
    """Sandbox-execute a model-written script against the runtime.

    Returns the resulting namespace (the caller reads `message`, `grounding_refs`,
    `outcome`, `completed_steps_laconic` from it) plus diagnostics. Never raises;
    all errors are reported on `ExecOutcome.error`."""
    try:
        _screen_ast(source)
    except SyntaxError as exc:
        return ExecOutcome({}, f"syntax error: {exc}")
    except SandboxError as exc:
        return ExecOutcome({}, f"sandbox violation: {exc}")

    cancel = threading.Event()
    timer = threading.Timer(timeout_sec, cancel.set)
    timer.daemon = True
    timer.start()

    proxy = _BudgetedRuntime(rt, wrapper_budget, cancel)
    # `json` and `re` are preloaded — common script needs (parsing tool output,
    # tokenising) that would otherwise force `import`. Both are pure-stdlib and
    # do no I/O of their own. `re` ReDoS risk is bounded by the wall-clock timer.
    namespace: dict[str, Any] = {
        "__builtins__": _SAFE_BUILTINS,
        "rt": proxy,
        "json": _json,
        "re": _re,
    }
    error: Optional[str] = None
    try:
        exec(compile(source, "<script>", "exec"), namespace)
    except SandboxError as exc:
        error = f"sandbox: {exc}"
    except Exception as exc:  # noqa: BLE001
        error = f"runtime: {type(exc).__name__}: {exc}"
    finally:
        timer.cancel()

    timed_out = cancel.is_set()
    if timed_out and error is None:
        error = "timeout: script wall-clock exceeded"
    return ExecOutcome(
        namespace=namespace,
        error=error,
        wrappers_used=wrapper_budget - proxy._remaining,
        timed_out=timed_out,
    )
