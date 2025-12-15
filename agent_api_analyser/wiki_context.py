import json
from datetime import datetime
from pathlib import Path
from typing import Iterable, Literal, List

from openai import OpenAI
from pydantic import BaseModel, Field


def _extract_summary(text: str, max_lines: int = 120, max_chars: int = 1500) -> str:
    """Grab headings, bullets, and early paragraphs to keep context compact."""
    lines = text.splitlines()
    picked: list[str] = []
    total = 0
    for ln in lines:
        stripped = ln.strip()
        if stripped.startswith("#") or stripped.startswith("-") or stripped.startswith("*") or stripped == "":
            picked.append(ln)
        elif len(picked) < max_lines:
            picked.append(ln)
        total += len(ln)
        if len(picked) >= max_lines or total >= max_chars:
            break
    return "\n".join(picked)


def build_wiki_context(
    docs_root: str | Path = "sgr-knowledge-agent-erc3_test/docs",
    max_total_chars: int = 8000,
    include_index_meta: bool = True,
) -> str:
    """
    Assemble a compact wiki context from downloaded pages and wiki_index.json.
    Falls back to scanning markdown files if index is missing.
    """
    root = Path(docs_root)
    index_path = root / "wiki_index.json"

    files: Iterable[dict]
    index_meta = ""
    if index_path.exists():
        idx = json.loads(index_path.read_text(encoding="utf-8"))
        files = idx.get("files", [])
        if include_index_meta:
            index_meta = f"wiki tree sha1: {idx.get('tree_sha1')}, generated_at: {idx.get('generated_at')}"
    else:
        files = [{"path": str(p.relative_to(root))} for p in sorted(root.rglob("*.md"))]

    chunks: list[str] = []
    total_chars = 0
    for entry in files:
        path = entry.get("path")
        if not path:
            continue
        file_path = root / path
        if not file_path.exists():
            continue
        content = file_path.read_text(encoding="utf-8")
        snippet = _extract_summary(content)
        chunk = f"## {path} (sha1={entry.get('sha1')}, size={entry.get('size_bytes')})\n{snippet}"
        total_chars += len(chunk)
        chunks.append(chunk)
        if total_chars >= max_total_chars:
            break

    prefix = "# Wiki context\n"
    if index_meta:
        prefix += index_meta + "\n\n"

    return prefix + "\n\n".join(chunks)


def _estimate_tokens(entry: dict) -> int:
    if "tokens" in entry:
        return int(entry["tokens"])
    size = entry.get("size_bytes")
    if size is None:
        return 0
    # rough estimate: 4 chars per token
    return max(1, int(size / 4))


def _score_path(path: str) -> tuple[int, str]:
    p = path.lower()
    scoring_table = [
        (("rule", "policy", "security"), 100, "rule/policy/security"),
        (("system", "api", "process"), 90, "systems/processes"),
        (("background", "mission", "vision", "strategy"), 80, "background/mission"),
        (("culture", "skills", "marketing"), 70, "culture/skills/marketing"),
        (("office", "offices"), 50, "offices"),
        (("people", "team"), 40, "people/team"),
    ]
    for kws, score, reason in scoring_table:
        if any(k in p for k in kws):
            return score, reason
    return 20, "misc"


def plan_wiki_reads(
    index_path: str | Path = "sgr-knowledge-agent-erc3_test/docs/wiki_index.json",
    context_budget_tokens: int = 32000,
    batch_size_tokens: int = 8000,
) -> list[dict]:
    """
    Build a ranked plan of wiki files:
    - priority: higher first
    - action: full | skim | batch (if file too large)
    - suggest_batch_with: group mates to send together when they fit
    - include_in_first_pass: True if cumulative tokens stay within budget
    """
    idx = json.loads(Path(index_path).read_text(encoding="utf-8"))
    files = idx.get("files", [])

    entries = []
    for f in files:
        path = f.get("path")
        if not path:
            continue
        tokens = _estimate_tokens(f)
        priority, reason = _score_path(path)
        action = "full"
        if tokens > batch_size_tokens:
            action = "batch"
        elif priority < 40:
            action = "skim"
        group = path.split("/")[0] if "/" in path else "root"
        entries.append(
            {
                "path": path,
                "tokens": tokens,
                "priority": priority,
                "reason": reason,
                "action": action,
                "group": group,
            }
        )

    # sort by priority desc, then shorter first
    entries.sort(key=lambda e: (-e["priority"], e["tokens"]))

    # cumulative budget flag
    cum = 0
    for e in entries:
        cum += e["tokens"]
        e["include_in_first_pass"] = cum <= context_budget_tokens

    # group batch suggestions
    groups: dict[str, list[dict]] = {}
    for e in entries:
        groups.setdefault(e["group"], []).append(e)

    for group_entries in groups.values():
        total = sum(g["tokens"] for g in group_entries)
        if total <= batch_size_tokens:
            grouped_paths = [g["path"] for g in group_entries]
            for g in group_entries:
                g["suggest_batch_with"] = [p for p in grouped_paths if p != g["path"]]
        else:
    for g in group_entries:
        g["suggest_batch_with"] = []

    return entries


# --- LLM-driven plan builder ---

class WikiPlanEntry(BaseModel):
    path: str
    priority: int = Field(..., description="higher means read earlier")
    action: Literal["full", "skim", "batch"]
    reason: str
    include_in_first_pass: bool
    suggest_batch_with: List[str] = Field(default_factory=list)
    note: str | None = None


class WikiPlan(BaseModel):
    generated_at: str
    context_budget_tokens: int
    batch_size_tokens: int
    entries: List[WikiPlanEntry]
    strategy: str | None = None


def propose_wiki_plan(
    wiki_index_path: str | Path = "sgr-knowledge-agent-erc3_test/docs/wiki_index.json",
    context_budget_tokens: int = 32000,
    batch_size_tokens: int = 8000,
    model: str = "gpt-4o-mini",
    out_dir: str | Path = "agent_api_analyser/api-report/context",
) -> Path:
    """
    Ask LLM to rank/group wiki files based on metadata. Saves plan JSON to out_dir with timestamp.
    """
    idx = json.loads(Path(wiki_index_path).read_text(encoding="utf-8"))
    files = idx.get("files", [])

    class LlmPlan(BaseModel):
        plan: WikiPlan

    client = OpenAI()
    now = datetime.utcnow().isoformat()

    system = (
        "You are planning how to read wiki files to build a concise context for an API scenario generator. "
        "Use the provided file metadata (path, tokens, size_bytes, sha1). "
        "Prioritize files with security/policies/rules, systems/processes/API usage, business goals, then people/offices. "
        "Decide action per file: full (read fully), skim (check briefly), batch (split or send with group if large). "
        "Group related files that fit together under the batch_size_tokens. "
        "Respect context_budget_tokens for include_in_first_pass flag."
    )
    user = {
        "context_budget_tokens": context_budget_tokens,
        "batch_size_tokens": batch_size_tokens,
        "files": files,
    }

    resp = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user)},
        ],
        response_format=LlmPlan,
    )

    plan: WikiPlan = resp.choices[0].message.parsed.plan
    # stamp metadata if not set
    plan.generated_at = now
    plan.context_budget_tokens = context_budget_tokens
    plan.batch_size_tokens = batch_size_tokens

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    target = out_path / f"wiki_plan_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
    target.write_text(plan.model_dump_json(indent=2, ensure_ascii=False), encoding="utf-8")
    return target
