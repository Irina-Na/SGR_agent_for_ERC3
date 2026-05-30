"""Runtime multi-doc policy collection for ECOM tasks.

The collector is recall-oriented and bounded. It derives all task-specific
context from the current trial's discovered /docs tree, doc triggers, live doc
content, and task text. It intentionally does not encode benchmark filenames,
entity categories, city names, table names, or task-specific rules.
"""
from __future__ import annotations

import json
import re
from typing import Dict, Iterable, List, Literal, Tuple

from openai import OpenAI
from pydantic import BaseModel, Field

from api_tools import Req_Read
from ecom_discovery import SessionDiscovery


DOC_READ_LIMIT = 5
DOC_CONTEXT_CHAR_BUDGET = 24_000
DOC_SNIPPET_CHAR_LIMIT = 7_000


class PolicyDocSelection(BaseModel):
    path: str
    status: Literal["authoritative", "candidate", "rejected"]
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    matched_subjects: List[str] = Field(default_factory=list)
    matched_operations: List[str] = Field(default_factory=list)
    scope_notes: str = ""
    why_relevant: str = ""


class PolicyCollectionResult(BaseModel):
    authoritative_docs: List[PolicyDocSelection] = Field(default_factory=list)
    candidate_docs: List[PolicyDocSelection] = Field(default_factory=list)
    rejected_docs: List[PolicyDocSelection] = Field(default_factory=list)
    injected_context: str = ""
    reason: str = ""
    diagnostics: Dict[str, object] = Field(default_factory=dict)


class _SelectionPayload(BaseModel):
    selections: List[PolicyDocSelection] = Field(default_factory=list)
    reason: str = ""


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "for", "from",
    "has", "have", "how", "i", "if", "in", "is", "it", "me", "no", "not",
    "of", "on", "or", "our", "please", "should", "that", "the", "these",
    "this", "to", "we", "were", "what", "when", "where", "which", "with",
    "would", "you", "your",
}


def _tokens(text: str) -> set[str]:
    return {
        token for token in re.findall(r"[A-Za-z0-9]+", text.lower())
        if len(token) >= 3 and token not in _STOPWORDS
    }


def _extract_json(content: str) -> str:
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
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1].strip() if start != -1 and end != -1 and start < end else text


def rank_doc_paths(
    task_text: str,
    discovery: SessionDiscovery,
    runtime_context: str = "",
) -> List[Tuple[str, float, List[str]]]:
    """Deterministically rank docs by runtime text overlap.

    This is a generic retrieval heuristic, not a task rule. It uses only doc
    paths, LLM-discovered triggers, and the task text.
    """
    task_tokens = _tokens(task_text + "\n" + runtime_context)
    ranked: List[Tuple[str, float, List[str]]] = []
    for path in discovery.docs_tree or []:
        triggers = list(discovery.policy_doc_index.get(path, []) or [])
        path_text = path.replace("/", " ").replace("-", " ").replace("_", " ")
        path_tokens = _tokens(path_text)
        trigger_tokens = _tokens(" ".join(triggers))
        overlap = sorted(task_tokens & (path_tokens | trigger_tokens))
        score = float(len(overlap) * 3)
        score += sum(2.0 for token in trigger_tokens if token in task_tokens)
        if triggers:
            score += 0.5
        score += min(path.count("/"), 5) * 0.05
        ranked.append((path, score, overlap))
    ranked.sort(key=lambda item: (-item[1], item[0]))
    return ranked


def _read_doc(rt, path: str) -> str:
    try:
        return rt.read(Req_Read(tool="read", path=path)).content or ""
    except Exception:
        return ""


def _headings_and_relevant_lines(content: str, task_text: str, limit: int) -> str:
    task_tokens = _tokens(task_text)
    lines = content.splitlines()
    picked: List[str] = []
    for line in lines:
        if line.strip().startswith("#"):
            picked.append(line)
    for idx, line in enumerate(lines):
        if _tokens(line) & task_tokens:
            picked.extend(lines[max(0, idx - 2):min(len(lines), idx + 3)])
    if not picked:
        picked = lines[:80]

    out: List[str] = []
    seen = set()
    total = 0
    for line in picked:
        key = line.strip()
        if key in seen:
            continue
        seen.add(key)
        if total + len(line) + 1 > limit:
            break
        out.append(line)
        total += len(line) + 1
    return "\n".join(out).strip()


def _bounded_doc_text(content: str, task_text: str, limit: int = DOC_SNIPPET_CHAR_LIMIT) -> str:
    if len(content) <= limit:
        return content
    snippet = _headings_and_relevant_lines(content, task_text, limit)
    if snippet:
        return snippet + f"\n[doc truncated from {len(content)} chars]"
    return content[:limit] + f"\n[doc truncated from {len(content)} chars]"


def _build_injected_context(
    authoritative: List[PolicyDocSelection],
    candidates: List[PolicyDocSelection],
    read_docs: Dict[str, str],
    task_text: str,
) -> str:
    selected = [*authoritative, *candidates]
    if not selected:
        return ""

    chunks = [
        "# RUNTIME POLICY CONTEXT",
        (
            "Runtime-selected /docs material for this task. Authoritative docs "
            "may define required grain, filters, answer format, permission "
            "rules, or workflow rules. Candidate docs are relevant but may have "
            "scope qualifiers; do not assume candidate scopes apply "
            "automatically. Use them to discover required rules and verify "
            "against task/runtime evidence."
        ),
    ]
    remaining = DOC_CONTEXT_CHAR_BUDGET - sum(len(chunk) + 2 for chunk in chunks)
    for label, docs in (("AUTHORITATIVE DOCS", authoritative), ("CANDIDATE DOCS", candidates)):
        if not docs:
            continue
        section = f"\n## {label}"
        chunks.append(section)
        remaining -= len(section) + 2
        for selection in docs:
            content = read_docs.get(selection.path, "")
            header = (
                f"\n### {selection.path}\n"
                f"status: {selection.status}; confidence: {selection.confidence:.2f}; "
                f"scope: {selection.scope_notes or 'not classified'}\n"
                f"why: {selection.why_relevant or 'runtime-selected'}\n"
            )
            body_limit = max(0, min(DOC_SNIPPET_CHAR_LIMIT, remaining - len(header) - 64))
            if body_limit <= 0:
                break
            body = _bounded_doc_text(content, task_text, body_limit)
            block = header + body
            if len(block) > remaining:
                block = block[:remaining] + "\n[policy context budget exhausted]"
            chunks.append(block)
            remaining -= len(block) + 2
            if remaining <= 0:
                break
        if remaining <= 0:
            break
    return "\n".join(chunks).strip()


def _build_collection(
    selections: List[PolicyDocSelection],
    read_docs: Dict[str, str],
    task_text: str,
    reason: str,
    llm_failed: bool = False,
) -> PolicyCollectionResult:
    authoritative = [s for s in selections if s.status == "authoritative"]
    candidates = [s for s in selections if s.status == "candidate"]
    rejected = [s for s in selections if s.status == "rejected"]
    injected = _build_injected_context(authoritative, candidates, read_docs, task_text)
    return PolicyCollectionResult(
        authoritative_docs=authoritative,
        candidate_docs=candidates,
        rejected_docs=rejected,
        injected_context=injected,
        reason=reason,
        diagnostics={
            "docs_read": len(read_docs),
            "llm_failed": llm_failed,
            "context_chars": len(injected),
        },
    )


def _fallback_collection(
    read_docs: Dict[str, str],
    ranked: List[Tuple[str, float, List[str]]],
    task_text: str,
    reason: str,
) -> PolicyCollectionResult:
    rank_by_path = {path: (score, overlap) for path, score, overlap in ranked}
    candidates: List[PolicyDocSelection] = []
    for path in read_docs:
        score, overlap = rank_by_path.get(path, (0.0, []))
        confidence = max(0.25, min(0.75, score / 12.0 if score else 0.35))
        candidates.append(PolicyDocSelection(
            path=path,
            status="candidate",
            confidence=confidence,
            matched_subjects=overlap,
            matched_operations=[],
            scope_notes="runtime fallback; scope not classified",
            why_relevant="Selected by deterministic runtime ranking.",
        ))
    return _build_collection(candidates, read_docs, task_text, reason, llm_failed=True)


def _filter_known_selections(
    selections: Iterable[PolicyDocSelection],
    read_docs: Dict[str, str],
    ranked: List[Tuple[str, float, List[str]]],
) -> List[PolicyDocSelection]:
    known = set(read_docs)
    rank_by_path = {path: (score, overlap) for path, score, overlap in ranked}
    out: List[PolicyDocSelection] = []
    seen = set()
    for selection in selections:
        if selection.path not in known or selection.path in seen:
            continue
        seen.add(selection.path)
        out.append(selection)

    rejected = {selection.path for selection in out if selection.status == "rejected"}
    selected = {selection.path for selection in out}
    for path in read_docs:
        if path in selected or path in rejected:
            continue
        score, overlap = rank_by_path.get(path, (0.0, []))
        if score <= 0:
            continue
        out.append(PolicyDocSelection(
            path=path,
            status="candidate",
            confidence=max(0.25, min(0.65, score / 14.0)),
            matched_subjects=overlap,
            matched_operations=[],
            scope_notes="LLM omitted this runtime-ranked doc; retained as candidate.",
            why_relevant="Runtime ranking found task/doc token overlap.",
        ))
    return out


_COLLECTOR_SYSTEM = """You classify runtime-read /docs documents for a task.

You are given the user task and bounded content from documents selected by
runtime discovery and ranking. Classify each document path as:
- authoritative: the document contains a CONCRETE RULE that determines this
  task's answer — a required grain, filter, scope qualifier, output format,
  permission/security/payment/mutation constraint, or workflow step that would
  change the answer if followed vs. ignored. The rule must address THIS task's
  subject AND scope, not just the general topic area.
- candidate: the document is semantically relevant but scope or applicability is
  uncertain, or it provides supporting context without a decisive rule. Scoped
  docs should usually be candidate, not rejected, unless the scope clearly
  contradicts the task.
- rejected: the document is clearly unrelated or contradicted by the task.

Rules:
- Do not invent paths.
- Do not require every scope qualifier to appear in the task. Unknown scope is a
  reason for candidate, not rejection.
- Prefer recall over precision, bounded by the provided documents.
- Match on meaning; filenames, folders, extensions, roles, tools, and table names
  may rotate.
- General index/overview/README/landing documents are NOT authoritative. They
  describe the environment but do not define answer-determining rules for a
  specific task. Classify them as candidate (if their content actually helps
  this task) or rejected (if not). A doc that merely mentions the task's topic
  area in passing is not authoritative — only a doc whose rule, if applied,
  would change the numeric or categorical answer.
- If no document contains such a decisive rule, return zero authoritative docs.
  Zero authoritative is the correct answer for many tasks; do not promote a
  weak match to fill the slot.
- Paths and identifiers found inside doc content are documentation, not ground
  truth — never treat example paths or names from a doc as confirmed to exist.

Return only one JSON object, no markdown/prose, matching this schema:
"""


def collect_task_policies(
    task_text: str,
    discovery: SessionDiscovery,
    rt,
    llm_client: OpenAI,
    model: str,
    runtime_context: str = "",
    doc_read_limit: int = DOC_READ_LIMIT,
) -> PolicyCollectionResult:
    """Read, classify, and format relevant runtime policy docs for a task."""
    docs = list(discovery.docs_tree or [])
    if not docs:
        return PolicyCollectionResult(reason="no /docs files discovered")

    ranked = rank_doc_paths(task_text, discovery, runtime_context)
    to_read = [path for path, _score, _overlap in ranked[:max(1, doc_read_limit)]]
    read_docs = {path: _read_doc(rt, path) for path in to_read}
    read_docs = {path: content for path, content in read_docs.items() if content}
    if not read_docs:
        return PolicyCollectionResult(
            reason="no ranked /docs content could be read",
            diagnostics={"docs_read": 0, "llm_failed": False, "context_chars": 0},
        )

    doc_payload = []
    for path, content in read_docs.items():
        score, overlap = next(((s, o) for p, s, o in ranked if p == path), (0.0, []))
        doc_payload.append({
            "path": path,
            "runtime_rank_score": score,
            "runtime_overlap_terms": overlap,
            "triggers": discovery.policy_doc_index.get(path, []),
            "content": _bounded_doc_text(content, task_text, DOC_SNIPPET_CHAR_LIMIT),
        })

    schema = json.dumps(_SelectionPayload.model_json_schema(), ensure_ascii=False)
    user = (
        f"# Task\n{task_text}\n\n"
        f"# Current runtime context (date/identity/tool output)\n{runtime_context or '(none)'}\n\n"
        "# Runtime-read /docs candidates\n"
        f"{json.dumps(doc_payload, ensure_ascii=False, indent=2)}\n"
    )
    try:
        resp = llm_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _COLLECTOR_SYSTEM + schema},
                {"role": "user", "content": user},
            ],
            max_completion_tokens=4096,
        )
        content = resp.choices[0].message.content or ""
        payload = _SelectionPayload.model_validate_json(_extract_json(content))
        selections = _filter_known_selections(payload.selections, read_docs, ranked)
        if not selections:
            return _fallback_collection(read_docs, ranked, task_text, "LLM selected no usable docs")
        return _build_collection(
            selections,
            read_docs,
            task_text,
            payload.reason or "runtime policy collection completed",
        )
    except Exception as exc:  # noqa: BLE001
        return _fallback_collection(read_docs, ranked, task_text, f"policy collector LLM failed: {exc}")


def format_policy_collection_diag(
    collection: PolicyCollectionResult,
    catalog_size: int,
    drift: int = 0,
) -> str:
    selected = [
        *(doc.path for doc in collection.authoritative_docs),
        *(doc.path for doc in collection.candidate_docs),
    ]
    paths_field = ",".join(selected) if selected else "-"
    reason = (collection.reason or "").replace("\n", " ").replace("\r", " ").strip()
    if len(reason) > 200:
        reason = reason[:197] + "..."
    return (
        f"POLICY DIAG: catalog={catalog_size} drift={drift:+d} "
        f"read={collection.diagnostics.get('docs_read', 0)} "
        f"authoritative={len(collection.authoritative_docs)} "
        f"candidates={len(collection.candidate_docs)} "
        f"rejected={len(collection.rejected_docs)} "
        f"paths=[{paths_field}] reason=\"{reason}\""
    )
