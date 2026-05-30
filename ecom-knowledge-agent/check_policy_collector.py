"""Deterministic regression checks for runtime policy collection.

No network calls. The fake runtime records reads and returns synthetic /docs
content; fake LLM clients return fixed classifier JSON or raise.
"""
from __future__ import annotations

import json
import sys
from types import SimpleNamespace

from bitgn.vm.ecom.ecom_pb2 import NodeKind

from ecom_discovery import SessionDiscovery
from code_executor import run_script
from policy_collector import collect_task_policies
from task_framing import refresh_docs_for_trial


class FakeRuntime:
    def __init__(self, docs: dict[str, str], tree_paths: list[str] | None = None):
        self.docs = docs
        self.tree_paths = tree_paths or list(docs)
        self.reads: list[str] = []

    def read(self, cmd):
        self.reads.append(cmd.path)
        return SimpleNamespace(content=self.docs.get(cmd.path, ""))

    def tree(self, _cmd):
        root = SimpleNamespace(name="docs", kind=NodeKind.NODE_KIND_DIR, children=[])
        dirs: dict[str, object] = {"/docs": root}
        for path in self.tree_paths:
            parts = path.strip("/").split("/")
            cur_path = "/docs"
            cur = root
            for part in parts[1:-1]:
                cur_path = cur_path + "/" + part
                if cur_path not in dirs:
                    node = SimpleNamespace(name=part, kind=NodeKind.NODE_KIND_DIR, children=[])
                    cur.children.append(node)
                    dirs[cur_path] = node
                cur = dirs[cur_path]
            cur.children.append(SimpleNamespace(
                name=parts[-1], kind=NodeKind.NODE_KIND_FILE, children=[],
            ))
        return SimpleNamespace(root=root)


class MissingPathRuntime:
    paths = set()

    def read(self, _cmd):
        raise RuntimeError("read failed: not found")

    def list(self, _cmd):
        raise RuntimeError("list failed: not found")

    def tree(self, _cmd):
        raise RuntimeError("tree failed: not found")


class FakeLLM:
    def __init__(self, payload: dict | None = None, fail: bool = False):
        self.payload = payload or {"selections": [], "reason": ""}
        self.fail = fail
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **_kwargs):
        if self.fail:
            raise RuntimeError("forced classifier failure")
        msg = SimpleNamespace(content=json.dumps(self.payload))
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def base_discovery(paths: list[str]) -> SessionDiscovery:
    return SessionDiscovery(
        docs_tree=paths,
        policy_doc_index={
            path: path.replace("/", " ").replace("-", " ").split()
            for path in paths
        },
    )


def assert_true(condition: bool, label: str) -> None:
    if not condition:
        raise AssertionError(label)


def test_scoped_relevant_doc_is_candidate() -> None:
    path = "/docs/current-updates/count-led-bulbs-vienna.md"
    rt = FakeRuntime({path: "# LED bulb counting\nUse distinct catalogue SKUs in Vienna."})
    discovery = base_discovery([path])
    result = collect_task_policies(
        "How many catalogue products are LED Bulb?",
        discovery,
        rt,
        FakeLLM(fail=True),
        "fake",
    )
    assert_true(result.candidate_docs and result.candidate_docs[0].path == path, "scoped doc not retained")
    assert_true("# RUNTIME POLICY CONTEXT" in result.injected_context, "context not injected")


def test_authoritative_doc_from_classifier() -> None:
    path = "/docs/rules/counting.md"
    rt = FakeRuntime({path: "# Counting\nAnswer format and grain are defined here."})
    discovery = base_discovery([path])
    result = collect_task_policies(
        "How many catalogue products match this count report?",
        discovery,
        rt,
        FakeLLM({
            "selections": [{
                "path": path,
                "status": "authoritative",
                "confidence": 0.9,
                "matched_subjects": ["catalogue"],
                "matched_operations": ["count"],
                "scope_notes": "defines grain",
                "why_relevant": "defines count report grain",
            }],
            "reason": "doc defines answer grain",
        }),
        "fake",
    )
    assert_true(result.authoritative_docs and result.authoritative_docs[0].path == path, "authoritative missing")


def test_rejected_doc_not_injected() -> None:
    path = "/docs/returns.md"
    rt = FakeRuntime({path: "# Returns\nRefund workflow only."})
    discovery = base_discovery([path])
    result = collect_task_policies(
        "How many catalogue products are LED Bulb?",
        discovery,
        rt,
        FakeLLM({
            "selections": [{
                "path": path,
                "status": "rejected",
                "confidence": 0.95,
                "matched_subjects": [],
                "matched_operations": [],
                "scope_notes": "unrelated",
                "why_relevant": "returns only",
            }],
            "reason": "unrelated",
        }),
        "fake",
    )
    assert_true(not result.injected_context, "rejected doc was injected")
    assert_true(result.rejected_docs and result.rejected_docs[0].path == path, "rejection missing")


def test_refresh_docs_removes_stale_and_adds_new() -> None:
    old_path = "/docs/old.md"
    new_path = "/docs/new.md"
    discovery = base_discovery([old_path])
    refreshed = refresh_docs_for_trial(FakeRuntime({new_path: "new"}, [new_path]), discovery)
    assert_true(refreshed.docs_tree == [new_path], "docs tree did not refresh")
    assert_true(old_path not in refreshed.policy_doc_index, "stale trigger survived")
    assert_true(new_path in refreshed.policy_doc_index, "new path missing")


def test_inspect_tolerates_missing_optional_paths() -> None:
    script = """
print(len(rt.list('/missing').entries))
print(rt.read('/missing').content == '')
print(len(rt.tree('/missing', level=2).root.children))
"""
    outcome = run_script(
        script,
        MissingPathRuntime(),
        timeout_sec=5,
        wrapper_budget=10,
        tolerate_not_found=True,
    )
    assert_true(outcome.ok, f"tolerant inspect failed: {outcome.error}")
    assert_true("0\nTrue\n0" in outcome.captured_stdout, "missing paths not returned as empty")


def main() -> int:
    tests = [
        test_scoped_relevant_doc_is_candidate,
        test_authoritative_doc_from_classifier,
        test_rejected_doc_not_injected,
        test_refresh_docs_removes_stale_and_adds_new,
        test_inspect_tolerates_missing_optional_paths,
    ]
    for test in tests:
        test()
        print(f"OK {test.__name__}")
    print("OK: policy collector checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
