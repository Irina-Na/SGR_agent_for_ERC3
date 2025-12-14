"""
Security policy planner and extractor.

- Loads wiki index (or triggers fetch_wiki.py if missing).
- Asks LLM to produce a plan for which wiki files to read first and how to batch them.
- Executes that plan: reads wiki markdown, batches if large, and asks LLM to extract
  structured security/access rules.
- Saves plan and extracted policies under unique folders in agent_security_analyser/plans
  and agent_security_analyser/policy.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Literal, Optional

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

try:
    import tiktoken
except ImportError:
    tiktoken = None


DEFAULT_INDEX = Path("sgr-knowledge-agent-erc3_test/docs/wiki_index.json")
DEFAULT_DOCS_ROOT = Path("sgr-knowledge-agent-erc3_test/docs")
DEFAULT_FETCH_SCRIPT = Path("sgr-knowledge-agent-erc3_test/fetch_wiki.py")
PLANS_ROOT = Path("agent_security_analyser/plans")
POLICY_ROOT = Path("agent_security_analyser/policy")

# Load environment (e.g., OPENAI_API_KEY from repo root .env)
load_dotenv()


# --- Models for plan generation ---


class PlanGroup(BaseModel):
    priority: int = Field(1, description="1 is highest priority")
    strategy: Literal["skip", "together", "single", "batch"] = Field(
        "single", description="How to send content to LLM"
    )
    path_s: str | List[str] = Field(
        ...,
        description="If 'skip', 'single' or 'batch' - str, if 'together' - list the paths List[str]",
    )
    estimated_tokens: int | None = None


class PlanPayload(BaseModel):

    reason: str
    groups: List[PlanGroup]
    notes: str | None = None


# --- Models for policy extraction ---


class PolicyRule(BaseModel):
    name: str = Field(..., description="Short identifier for the rule")
    summary: str = Field(..., description="1-2 line essence of the rule")
    allow: List[str] = Field(default_factory=list, description="Explicitly permitted actions/conditions")
    deny: List[str] = Field(default_factory=list, description="Explicitly blocked actions/conditions")
    clarify: List[str] = Field(default_factory=list, description="When to ask for more detail")
    scope: List[str] = Field(default_factory=list, description="Roles/levels/resources/locations constraints")
    sources: List[str] = Field(..., min_length=10, description="Wiki file paths used (non-empty)")


class PolicyExtraction(BaseModel):
    policies: List[PolicyRule]
    gaps_or_questions: List[str] | None


# --- Helpers ---

def unique_run_id(prefix: str) -> str:
    """Deterministic timestamp-based id (no LLM influence)."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{ts}"


def ensure_index(index_path: Path = DEFAULT_INDEX, fetch_script: Path = DEFAULT_FETCH_SCRIPT) -> dict:
    if index_path.exists():
        return json.loads(index_path.read_text(encoding="utf-8"))
    if not fetch_script.exists():
        raise FileNotFoundError(f"wiki index missing and fetch script not found: {fetch_script}")
    subprocess.run([sys.executable, str(fetch_script)], check=True)
    if not index_path.exists():
        raise FileNotFoundError(f"wiki index still missing after fetch: {index_path}")
    return json.loads(index_path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_plan(plan_id: str, plan: PlanPayload, plans_root: Path = PLANS_ROOT) -> Path:
    plan_dir = plans_root / plan_id
    plan_dir.mkdir(parents=True, exist_ok=True)
    plan_path = plan_dir / "plan.json"
    save_json(plan_path, {"plan_id": plan_id, **plan.model_dump()})
    return plan_path


def load_plan(plan_path: Path) -> tuple[str, PlanPayload]:
    data = load_json(plan_path)
    plan_id = data.get("plan_id") or plan_path.parent.name
    payload = {k: v for k, v in data.items() if k != "plan_id"}
    return plan_id, PlanPayload(**payload)


def save_policies(plan_id: str, policies: List[PolicyRule], policy_root: Path = POLICY_ROOT) -> Path:
    policy_dir = policy_root / plan_id
    policy_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "plan_id": plan_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "policies": [p.model_dump() for p in policies],
    }
    policy_path = policy_dir / "policies.json"
    save_json(policy_path, payload)
    return policy_path


def make_file_batches(
    paths: List[str],
    texts: List[str],
    tokens_by_path: dict[str, int],
    max_batch_tokens: int,
) -> List[tuple[List[str], str]]:
    """
    Batch files by path-level token estimates. Falls back to text token count if missing.
    Returns list of (paths_in_batch, combined_text).
    """
    batches: List[tuple[List[str], str]] = []
    current_paths: List[str] = []
    current_texts: List[str] = []
    current_tokens = 0

    for rel_path, text in zip(paths, texts):
        tok_val = tokens_by_path.get(rel_path)
        tok = tok_val if isinstance(tok_val, int) and tok_val > 0 else 0
        if current_paths and current_tokens + tok > max_batch_tokens:
            batches.append((current_paths, "\n\n".join(current_texts)))
            current_paths, current_texts, current_tokens = [], [], 0
        current_paths.append(rel_path)
        current_texts.append(text)
        current_tokens += tok

    if current_paths:
        batches.append((current_paths, "\n\n".join(current_texts)))
    return batches


# --- Core planner/extractor ---


class SecurityPolicyPlanner:
    def __init__(self, model: str = "gpt-4.1") -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate_plan(
        self,
        wiki_index: dict,
        plan_id: Optional[str] = None,
        max_batch_tokens: int = 6000,
        docs_root: Path = DEFAULT_DOCS_ROOT,
        model: str = "gpt-5.1"
    ) -> tuple[str, PlanPayload]:
        files = wiki_index.get("files", [])
        summary = [
            {
                "path": f.get("path"),
                "tokens": f.get("tokens"),
            }
            for f in files
            if f.get("path")
        ]
        plan_id_to_use = plan_id or unique_run_id("plan")
        readme_path = docs_root / "README.md"
        readme_content = readme_path.read_text(encoding="utf-8")
        messages = [
            {
                "role": "system",
                "content": (
                    "You rank wiki files to extract security/access policies. "
                    "Use filename/size/token hints to set priority. "
                    "Strategies: "
                    "skip not important files"
                    "'single' (one file per LLM call),"
                    "'batch' (split large files into chunks under the provided token limit),"
                    "'together' (send listed files together)."
                    "Prefer files with names like rule/policy/security/access/guardrail/playbook."
                    "Mark only real paths from the index."
                ),
            },
            {
                "role": "user",
                 "content": (
                        f"Token limit per batch: {max_batch_tokens}.\n"
                        f"Wiki index entries (path, tokens, size_bytes, sha1):\n"
                        f"{json.dumps(summary, ensure_ascii=False, indent=2)}\n\n"
                        f"\n\nWiki README content (full):\n{readme_content}"
                        "Return a plan with prioritized groups."
                    ),
            },
        ]
        resp = self.client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            response_format=PlanPayload,
        )
        payload: PlanPayload = resp.choices[0].message.parsed
        return plan_id_to_use, payload

    def extract_policies_for_chunk(
        self,
        content: str,
        sources: List[str],
        max_completion_tokens: int = 2000,
    ) -> PolicyExtraction:
        messages = [
            {
                "role": "system",
                "content": (
                    "Extract security/access/data-protection policies into PolicyRule items:\n"
                    "- name: short identifier\n"
                    "- summary: 1-2 line essence of the rule\n"
                    "- allow: explicitly permitted actions/conditions (can be empty)\n"
                    "- deny: explicitly blocked actions/conditions (can be empty)\n"
                    "- clarify: when to ask for more detail (can be empty)\n"
                    "- scope: roles/levels/resources/locations constraints\n"
                    "- sources: wiki paths you used\n"
                    "Only include the most important rules. Do not invent IDs. Leave lists empty if not applicable."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Sources: {sources}\n"
                    "Content:\n" + content
                ),
            },
        ]
        resp = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=messages,
            response_format=PolicyExtraction,
            max_completion_tokens=max_completion_tokens,
        )
        return resp.choices[0].message.parsed

    def run_plan(
        self,
        plan: PlanPayload,
        docs_root: Path = DEFAULT_DOCS_ROOT,
        wiki_index: Optional[dict] = None,
        max_batch_tokens: int = 6000,
    ) -> List[PolicyRule]:
        tokens_by_path = {}
        if wiki_index:
            tokens_by_path = {
                f.get("path"): f.get("tokens") for f in wiki_index.get("files", []) if f.get("path")
            }
        collected: List[PolicyRule] = []
        for group in sorted(plan.groups, key=lambda g: g.priority):
            paths = group.path_s if isinstance(group.path_s, list) else [group.path_s]
            if not paths:
                continue
            texts: List[str] = []
            for rel_path in paths:
                file_path = docs_root / rel_path
                if not file_path.exists():
                    continue
                content = file_path.read_text(encoding="utf-8")
                texts.append(f"### {rel_path}\n{content}")

            if not texts:
                continue

            if group.strategy == "single":
                for rel_path, text in zip(paths, texts):
                    extraction = self.extract_policies_for_chunk(text, [rel_path])
                    collected.extend(extraction.policies)
                continue

            batches = make_file_batches(
                paths,
                texts,
                tokens_by_path,
                max_batch_tokens,
            )

            for batch_paths, batch_text in batches:
                extraction = self.extract_policies_for_chunk(batch_text, batch_paths)
                collected.extend(extraction.policies)

        return collected


def materialize_plan_and_policies(
    index_path: Path = DEFAULT_INDEX,
    docs_root: Path = DEFAULT_DOCS_ROOT,
    plans_root: Path = PLANS_ROOT,
    policy_root: Path = POLICY_ROOT,
    max_batch_tokens: int = 6000,
    model: str = "gpt-4o-mini",
) -> tuple[Path, Path]:
    """
    High-level helper:
    - ensure wiki index exists,
    - ask LLM for a plan,
    - save plan under plans/<id>/plan.json,
    - run plan to extract policies,
    - save policies under policy/<id>/policies.json.
    Returns (plan_path, policy_path).
    """
    wiki_index = ensure_index(index_path=index_path)
    planner = SecurityPolicyPlanner(model=model)
    plan_id, plan = planner.generate_plan(
        wiki_index,
        max_batch_tokens=max_batch_tokens,
        docs_root=docs_root,
    )

    plan_path = save_plan(plan_id, plan, plans_root=plans_root)

    policies = planner.run_plan(
        plan,
        docs_root=docs_root,
        wiki_index=wiki_index,
        max_batch_tokens=max_batch_tokens,
    )
    policy_path = save_policies(plan_id, policies, policy_root=policy_root)
    return plan_path, policy_path


__all__ = [
    "PlanPayload",
    "PlanGroup",
    "PolicyRule",
    "PolicyExtraction",
    "SecurityPolicyPlanner",
    "materialize_plan_and_policies",
    "ensure_index",
    "load_json",
    "PLANS_ROOT",
    "POLICY_ROOT",
]
