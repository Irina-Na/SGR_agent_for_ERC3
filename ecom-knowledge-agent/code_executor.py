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
import collections as _collections
import datetime as _datetime
import io as _io
import json as _json
import re as _re
import threading
import typing as _typing
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Optional

from connectrpc.errors import ConnectError

from api_tools import (
    Req_Exec, Req_Find, Req_List, Req_Read, Req_Search, Req_Tree,
)


class SandboxError(Exception):
    """Sandbox-detected violation (AST, budget, timeout, runtime)."""


# --- AST screen ---------------------------------------------------------------

_BANNED_NAMES = frozenset({
    "eval", "exec", "compile", "open", "input",
    # `__import__` is provided via a restricted callable in _SAFE_BUILTINS so
    # the import *statement* works for preloaded modules (json/re). Bare
    # `__import__('os')` is rejected at runtime by that restricted callable.
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


_PRELOADED_MODULES = frozenset({"json", "re", "datetime", "collections", "typing"})


def _screen_ast(source: str) -> None:
    tree = ast.parse(source, mode="exec")
    # Top-level `return` at module scope is a SyntaxError at compile time with a
    # confusing message ("'return' outside function"). Models do this when they
    # think of the script as a function body. Catch it here with a clearer hint.
    for stmt in tree.body:
        if isinstance(stmt, ast.Return):
            raise SandboxError(
                "top-level `return` is not allowed — the script runs as a module, "
                "not a function. Assign to `message`, `grounding_refs`, `outcome`, "
                "`completed_steps_laconic` at module scope instead."
            )
    for node in ast.walk(tree):
        # Imports of PRELOADED modules (`import json`, `import re as r`, `from re
        # import search`) are harmless — the module is already in the script's
        # namespace and the `import` is a redundant no-op. Models keep writing
        # them out of habit despite the prompt; rejecting was pure friction.
        # All OTHER imports remain blocked (os, subprocess, etc.).
        if isinstance(node, ast.Import):
            offending = [a.name for a in node.names if a.name.split(".")[0] not in _PRELOADED_MODULES]
            if offending:
                raise SandboxError(f"imports not allowed (only json/re are preloaded): {offending}")
            continue
        if isinstance(node, ast.ImportFrom):
            mod = (node.module or "").split(".")[0]
            if mod not in _PRELOADED_MODULES:
                raise SandboxError(f"imports not allowed (only json/re are preloaded): from {node.module}")
            continue
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
    "iter", "next",                                         # iterator protocol
    "len", "list", "map", "max", "min", "pow", "range", "reversed", "round",
    "set", "slice", "sorted", "str", "sum", "tuple", "zip",
    "repr", "chr", "ord", "bin", "hex", "oct", "format", "print",
    "getattr", "hasattr",   # AST screen still rejects literal dunder/private names
    # Exception classes — honest scripts use try/except for fragile parsing.
    # These are types only; they perform no I/O.
    "Exception", "BaseException", "ValueError", "TypeError", "KeyError",
    "IndexError", "AttributeError", "StopIteration", "ArithmeticError",
    "ZeroDivisionError", "RuntimeError", "LookupError",
    "NameError", "UnicodeError", "UnicodeDecodeError", "UnicodeEncodeError",
    "AssertionError", "NotImplementedError", "OverflowError",
    "OSError", "FileNotFoundError", "NotADirectoryError", "IsADirectoryError",
    "PermissionError", "SystemExit", "GeneratorExit", "FloatingPointError",
)
_SAFE_BUILTINS: dict[str, Any] = {n: getattr(_b, n) for n in _SAFE_BUILTIN_NAMES}
_SAFE_BUILTINS.update({"True": True, "False": False, "None": None})

# Restricted `__import__` so that the Python `import` statement actually works
# for preloaded modules — `import json`, `import re as r`, `from re import search`.
# Everything else raises ImportError. The model frequently writes `import json`
# out of habit even when told it's preloaded; this turns that habit into a no-op
# instead of a fatal sandbox violation.
_ALLOWED_IMPORTS: dict[str, Any] = {
    "json": _json, "re": _re, "datetime": _datetime,
    "collections": _collections, "typing": _typing,
}


def _restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".")[0]
    if root not in _ALLOWED_IMPORTS:
        raise ImportError(f"module not available in sandbox: {name!r}")
    return _ALLOWED_IMPORTS[root]


_SAFE_BUILTINS["__import__"] = _restricted_import


# --- SQL parameter binding ----------------------------------------------------

def _bind_value(v: Any) -> str:
    """Render a Python value as a SQL literal. Mirrors sqlite3 type adaptation
    closely enough for the queries scripts actually write: None→NULL, bool→0/1,
    numbers as-is, everything else single-quote-escaped string."""
    if v is None:
        return "NULL"
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, (int, float)):
        return repr(v)
    s = str(v).replace("'", "''")
    return f"'{s}'"


def _bind_params(query: str, params) -> str:
    """Substitute `?` placeholders in `query` with values from `params`, matching
    the sqlite3 cursor.execute(query, params) contract every model already knows.

    Models call `rt.sql("SELECT ... WHERE id = ?", (x,))` out of habit; without
    this they get a cryptic positional-args TypeError. Substitution happens
    BEFORE the SQL tool sees the string, so it's the same architecture as if
    the script had built the string itself with f-strings — just safer and
    matching the expected API.

    A `:name` style dict binding is also accepted: `rt.sql("... WHERE id = :id",
    {"id": x})`. Mismatched counts raise so the script gets a clear error."""
    if not params:
        return query
    if isinstance(params, dict):
        # :name substitution. Sort by name length desc so :foo doesn't match :foobar prefix.
        for name in sorted(params, key=len, reverse=True):
            query = query.replace(f":{name}", _bind_value(params[name]))
        return query
    if isinstance(params, (str, bytes)):
        # User probably meant a single-element tuple; refuse the ambiguous case.
        raise TypeError(
            "rt.sql params must be a tuple/list/dict, not a bare string. "
            "Use (value,) for a single ? placeholder."
        )
    seq = list(params)
    parts = query.split("?")
    if len(parts) - 1 != len(seq):
        raise ValueError(
            f"rt.sql: query has {len(parts) - 1} ? placeholders but got {len(seq)} params"
        )
    out = [parts[0]]
    for i, val in enumerate(seq):
        out.append(_bind_value(val))
        out.append(parts[i + 1])
    return "".join(out)


# --- budgeted runtime proxy ---------------------------------------------------

class _BudgetedRuntime:
    """Script-facing facade: primitive-arg methods over EcomRuntime, each
    decrementing the wrapper budget and checking the cancel signal. The
    underlying rt remains the only I/O surface, so evidence-ledger capture
    and the gate work identically to the step loop."""

    def __init__(
        self,
        rt,
        budget: int,
        cancel: threading.Event,
        tolerate_not_found: bool = False,
    ):
        # use slot-like attrs prefixed with _ so AST screen's dunder/private
        # block prevents the script from poking at them (rt._rt etc.).
        # (Even if it tried, the AST screen rejects underscore attrs above.)
        object.__setattr__(self, "_rt", rt)
        object.__setattr__(self, "_remaining", budget)
        object.__setattr__(self, "_cancel", cancel)
        object.__setattr__(self, "_tolerate_not_found", tolerate_not_found)

    def _check(self, name: str) -> None:
        if self._cancel.is_set():
            raise SandboxError("script wall-clock timeout exceeded")
        if self._remaining <= 0:
            raise SandboxError(f"wrapper-call budget exhausted at rt.{name}()")
        object.__setattr__(self, "_remaining", self._remaining - 1)

    def _not_found(self, exc: Exception) -> bool:
        if not self._tolerate_not_found:
            return False
        if isinstance(exc, ConnectError):
            return "not found" in (getattr(exc, "message", "") or "").lower()
        return "not found" in str(exc).lower()

    def _empty_read(self, path: str):
        return SimpleNamespace(content="", truncated=False, missing=True, path=path)

    def _empty_list(self, path: str):
        return SimpleNamespace(entries=[], truncated=False, missing=True, path=path)

    def _empty_tree(self, root: str):
        return SimpleNamespace(
            root=SimpleNamespace(name=(root or "/").rstrip("/") or "/", children=[]),
            truncated=False,
            missing=True,
        )

    @staticmethod
    def _with_missing(r):
        """Ensure the result exposes `.missing` (default False) so the model's
        `if r.missing` / `not r.missing` checks work uniformly across real
        protobuf responses and the empty placeholders above."""
        if r is None or hasattr(r, "missing"):
            return r
        return _ResultProxy(r, False)

    # one method per typed wrapper; primitive args (script can't construct Req_*)
    def read(self, path: str, number: bool = False, start_line: int = 0, end_line: int = 0):
        self._check("read")
        try:
            return self._rt.read(Req_Read(tool="read", path=path, number=number,
                                          start_line=start_line, end_line=end_line))
        except Exception as exc:
            if self._not_found(exc):
                return self._empty_read(path)
            raise

    def list(self, path: str = "/"):
        self._check("list")
        try:
            return self._rt.list(Req_List(tool="list", path=path))
        except Exception as exc:
            if self._not_found(exc):
                return self._empty_list(path)
            raise

    def tree(self, root: str = "", level: int = 2):
        self._check("tree")
        try:
            return self._rt.tree(Req_Tree(tool="tree", root=root, level=level))
        except Exception as exc:
            if self._not_found(exc):
                return self._empty_tree(root)
            raise

    def find(self, name: str, root: str = "/", kind: str = "all", limit: int = 10):
        self._check("find")
        _kind_norm = {
            "file": "files", "files": "files",
            "dir": "dirs", "directory": "dirs", "directories": "dirs", "dirs": "dirs",
            "all": "all", "any": "all", "": "all",
        }
        kind = _kind_norm.get((kind or "all").lower(), kind)
        try:
            resp = self._rt.find(Req_Find(tool="find", name=name, root=root,
                                          kind=kind, limit=min(limit, 20)))
        except Exception as exc:
            if self._not_found(exc):
                return []
            raise
        # Models routinely write `for m in rt.find(...)` and `len(rt.find(...))`.
        # FindResponse is neither iterable nor len-able; return the list directly.
        return list(getattr(resp, "matches", None) or getattr(resp, "entries", None) or [])

    def search(self, pattern: str, root: str = "/", limit: int = 10):
        self._check("search")
        try:
            resp = self._rt.search(Req_Search(tool="search", pattern=pattern,
                                              root=root, limit=min(limit, 20)))
        except Exception as exc:
            if self._not_found(exc):
                return []
            raise
        return list(getattr(resp, "matches", None) or [])

    def sql(self, query: str, params=()) -> list[dict]:
        self._check("sql")
        return self._rt.sql(_bind_params(query, params))

    def sql_raw(self, query: str, params=()) -> str:
        self._check("sql_raw")
        return self._rt.sql_raw(_bind_params(query, params))

    def exec(self, path: str, args=(), stdin: str = ""):
        self._check("exec")
        try:
            return self._rt.exec(Req_Exec(tool="exec", path=path,
                                          args=list(args), stdin=stdin))
        except Exception as exc:
            if self._not_found(exc):
                return SimpleNamespace(stdout="", stderr=str(getattr(exc, "message", exc)),
                                       exit_code=127, missing=True)
            raise

    @property
    def tools(self):
        specs = getattr(self._rt, "tool_specs", {}) or {}
        out = {}
        for name, spec in specs.items():
            if isinstance(spec, dict):
                out[name] = {
                    "path": spec.get("path", f"/bin/{name}"),
                    "classification": spec.get("classification", "unknown"),
                    "help": spec.get("help_text", ""),
                }
            else:
                out[name] = {
                    "path": getattr(spec, "path", f"/bin/{name}"),
                    "classification": getattr(spec, "classification", "unknown"),
                    "help": getattr(spec, "help_text", ""),
                }
        return out

    def tool_help(self, name: str) -> str:
        self._check("tool_help")
        return self._rt.tool_help(name)

    def run_tool(self, name: str, args=(), stdin: str = ""):
        self._check("run_tool")
        return self._rt.run_tool(name, args=args, stdin=stdin)

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
    captured_stdout: str = ""

    @property
    def ok(self) -> bool:
        return self.error is None


_STDOUT_CAP = 32 * 1024  # bytes-of-text cap on captured Inspect/Answer stdout


def _make_capture_print(buf: _io.StringIO):
    """Drop-in replacement for `print` that writes into `buf` (bounded) and
    silently truncates once the cap is hit. Mirrors `print()`'s sep/end/flush
    semantics so model scripts behave the same as if they were printing to the
    real stdout."""
    def _print(*args, sep: str = " ", end: str = "\n", file=None, flush: bool = False):
        text = sep.join(str(a) for a in args) + end
        remaining = _STDOUT_CAP - buf.tell()
        if remaining <= 0:
            return
        if len(text) > remaining:
            text = text[:remaining]
        buf.write(text)
    return _print


def run_script(
    source: str,
    rt,
    timeout_sec: float = 30.0,
    wrapper_budget: int = 60,
    tolerate_not_found: bool = False,
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

    proxy = _BudgetedRuntime(
        rt,
        wrapper_budget,
        cancel,
        tolerate_not_found=tolerate_not_found,
    )
    stdout_buf = _io.StringIO()
    # Per-call `print` writes into the local buffer instead of the real stdout
    # so the Inspect phase can pipe the model's observations into the next LLM
    # call. We swap it into a copy of _SAFE_BUILTINS to keep the module-level
    # whitelist read-only and reusable across runs.
    builtins_view = dict(_SAFE_BUILTINS)
    builtins_view["print"] = _make_capture_print(stdout_buf)
    # `json` and `re` are preloaded — common script needs (parsing tool output,
    # tokenising) that would otherwise force `import`. Both are pure-stdlib and
    # do no I/O of their own. `re` ReDoS risk is bounded by the wall-clock timer.
    namespace: dict[str, Any] = {
        "__builtins__": builtins_view,
        "rt": proxy,
        "json": _json,
        "re": _re,
        "datetime": _datetime,
        "collections": _collections,
        "typing": _typing,
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
        captured_stdout=stdout_buf.getvalue(),
    )
