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
    intents: List[str] = Field(
        default_factory=list,
        description="Canonical intents this rule affects (e.g., public_answer, salary_view, project_status_change)",
    )


class PolicyConstraint(BaseModel):
    intent: str = Field(..., description="Normalized intent id (e.g., salary_view, project_status_change)")
    roles: List[str] = Field(default_factory=list, description="Allowed roles/levels; empty means any authenticated")
    override_roles: List[str] = Field(
        default_factory=list, description="Roles that can override responsibility limits (e.g., exec)"
    )
    locations: List[str] = Field(default_factory=list, description="Allowed locations; use ['any'] if unrestricted")
    allow_if_exec: bool = Field(
        default=False, description="If true, exec/level1 may bypass stricter role/location checks for this intent"
    )
    needs_responsibility: bool = Field(
        default=False, description="Require requester to be responsible/lead/owner for the target resource"
    )
    needs_assignment: bool = Field(
        default=False, description="Require requester to be on the project/team (e.g., for time logging)"
    )
    needs_clarification: bool = Field(
        default=False, description="Require disambiguation (who/what) before acting for this intent"
    )
    sensitivity: Literal["public", "internal", "sensitive", "highly_sensitive"] = Field(
        "internal", description="Data classification for this intent"
    )
    sources: List[str] = Field(..., min_length=1, description="Wiki file paths used (non-empty)")
    rationale: str = Field(..., description="<=2 sentence why these constraints exist")


class PolicyExtraction(BaseModel):
    policies: List[PolicyRule]
    constraints: List[PolicyConstraint] = Field(default_factory=list)
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


def save_policies(
    plan_id: str,
    policies: List[PolicyRule],
    constraints: List[PolicyConstraint] | None = None,
    policy_root: Path = POLICY_ROOT,
) -> Path:
    policy_dir = policy_root / plan_id
    policy_dir.mkdir(parents=True, exist_ok=True)
    constraints = constraints or []
    payload = {
        "plan_id": plan_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "policies": [p.model_dump() for p in policies],
        "constraints": [c.model_dump() for c in constraints],
        "constraints_map": {c.intent: c.model_dump() for c in constraints if c.intent},
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
        model: str = "gpt-5.1",
        readme_path: Path | None = None,
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
        readme_to_use = readme_path if readme_path else docs_root / "README.md"
        readme_content = ""
        if readme_to_use.exists():
            readme_content = readme_to_use.read_text(encoding="utf-8")
        messages = [
            {
                "role": "system",
                "content": (
                    "You rank wiki files to extract security/access/data-handling policies. "
                    "Use filename/size/token hints to set priority. "
                    "Strategies: skip (only if clearly irrelevant), 'single' (one file), "
                    "'batch' (split large files under the token limit), 'together' (send listed files together). "
                    "ALWAYS include files that define roles/levels, locations/offices, systems/APIs, or data categories "
                    ". Do not mark them as skip even if the filename "
                    "does not contain policy/security keywords. Prefer rule/policy/security/guardrail/playbook files "
                    "as highest priority but keep supporting context grouped at lower priority if needed. "
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
                    "Extract security/access/data-protection policies into PolicyRule and PolicyConstraint items.\n"
                    "Canonical intents (use when relevant, do not invent outside this set unless explicitly present):\n"
                    "- public_answer, wiki_read, customer_read, project_read, project_status_change, project_team_change,\n"
                    "  time_entry_create, salary_view, salary_change, pii_view.\n"
                    "PolicyRule fields:\n"
                    "- name: short identifier\n"
                    "- summary: 1-2 line essence of the rule\n"
                    "- allow / deny / clarify / scope: as stated in the content\n"
                    "- intents: map the rule to relevant intents from the canonical set\n"
                    "- sources: wiki paths you used (non-empty)\n"
                    "PolicyConstraint fields (drive runtime enforcement):\n"
                    "- intent: canonical intent id\n"
                    "- roles: allowed roles/levels (e.g., exec, lead, core)\n"
                    "- override_roles: roles that can bypass responsibility checks (e.g., exec)\n"
                    "- locations: allowed locations/offices or ['any']\n"
                    "- allow_if_exec: true when exec/level1 can override stricter checks\n"
                    "- needs_responsibility: true when requester must be responsible/lead/owner for target\n"
                    "- needs_assignment: true when requester must be on the project/team (e.g., time logging)\n"
                    "- needs_clarification: true when the rule requires disambiguation before acting\n"
                    "- sensitivity: public/internal/sensitive/highly_sensitive as implied by data classification\n"
                    "- sources: wiki paths you used\n"
                    "- rationale: short why this constraint exists\n"
                    "Ground everything in the provided content; do not hallucinate roles or permissions. "
                    "If a field is not specified by the content, leave it empty/false rather than guessing."
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
        verbose: bool = False,
    ) -> tuple[List[PolicyRule], List[PolicyConstraint]]:
        tokens_by_path = {}
        if wiki_index:
            tokens_by_path = {
                f.get("path"): f.get("tokens") for f in wiki_index.get("files", []) if f.get("path")
            }
        collected_policies: List[PolicyRule] = []
        collected_constraints: List[PolicyConstraint] = []
        for group in sorted(plan.groups, key=lambda g: g.priority):
            if verbose:
                print(
                    f"[run_plan] priority={group.priority} strategy={group.strategy} paths={group.path_s}",
                    flush=True,
                )
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
                    collected_policies.extend(extraction.policies)
                    collected_constraints.extend(extraction.constraints)
                continue

            batches = make_file_batches(
                paths,
                texts,
                tokens_by_path,
                max_batch_tokens,
            )

            for batch_paths, batch_text in batches:
                extraction = self.extract_policies_for_chunk(batch_text, batch_paths)
                collected_policies.extend(extraction.policies)
                collected_constraints.extend(extraction.constraints)

        return collected_policies, collected_constraints


def materialize_plan_and_policies(
    index_path: Path = DEFAULT_INDEX,
    docs_root: Path = DEFAULT_DOCS_ROOT,
    plans_root: Path = PLANS_ROOT,
    policy_root: Path = POLICY_ROOT,
    max_batch_tokens: int = 6000,
    model: str = "gpt-4o-mini",
    readme_path: Path | None = None,
    extra_policy_file: Path | None = None,
) -> tuple[Path, Path]:
    """
    High-level helper:
    - ensure wiki index exists,
    - ask LLM for a plan,
    - save plan under plans/<id>/plan.json,
    - run plan to extract policies,
    - optionally extract policies from an extra file (e.g., prep_desc.md),
    - save policies under policy/<id>/policies.json.
    Returns (plan_path, policy_path).
    """
    wiki_index = ensure_index(index_path=index_path)
    planner = SecurityPolicyPlanner(model=model)
    plan_id, plan = planner.generate_plan(
        wiki_index,
        max_batch_tokens=max_batch_tokens,
        docs_root=docs_root,
        readme_path=readme_path,
    )

    plan_path = save_plan(plan_id, plan, plans_root=plans_root)

    policies, constraints = planner.run_plan(
        plan,
        docs_root=docs_root,
        wiki_index=wiki_index,
        max_batch_tokens=max_batch_tokens,
        verbose=True,
    )

    if extra_policy_file and extra_policy_file.exists():
        extra_text = extra_policy_file.read_text(encoding="utf-8")
        extraction = planner.extract_policies_for_chunk(extra_text, [str(extra_policy_file)])
        policies.extend(extraction.policies)
        constraints.extend(extraction.constraints)

    policy_path = save_policies(plan_id, policies, constraints, policy_root=policy_root)
    return plan_path, policy_path


__all__ = [
    "PlanPayload",
    "PlanGroup",
    "PolicyRule",
    "PolicyConstraint",
    "PolicyExtraction",
    "SecurityPolicyPlanner",
    "materialize_plan_and_policies",
    "ensure_index",
    "load_json",
    "PLANS_ROOT",
    "POLICY_ROOT",
]
