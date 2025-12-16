"""
Minimal wiki policy extractor inspired by the succinct style of sgr-knowledge-agent-erc3_test/agent.py.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()

DEFAULT_DOCS_ROOT = Path("sgr-knowledge-agent-erc3_test/docs")
DEFAULT_INDEX = DEFAULT_DOCS_ROOT / "wiki_index.json"
FOUND_ROOT = Path("agent_security_analyser/agent_wiki_extraction/found_data")


class PolicyNotes(BaseModel):
    plan_id: str = Field(default="manual")
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source_found_path: str = ""
    notes: List[str]


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _client(model: str) -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return OpenAI(api_key=api_key), model


def extract_policy_notes(found_path: Path, docs_root: Path = DEFAULT_DOCS_ROOT, model: str = "gpt-4.1") -> Path:
    """
    Send policy-relevant files to LLM and capture concise notes.
    """
    found = FoundData.model_validate_json(found_path.read_text(encoding="utf-8"))
    policy_keys = {"roles_policies", "locations", "hierarchy_people", "systems", "actions_api", "public_content"}
    picks = [c for c in found.categories if c.key in policy_keys]

    bundle: List[dict] = []
    for cat in picks:
        for f in cat.files:
            content_path = docs_root / f.path
            if not content_path.exists():
                continue
            bundle.append(
                {
                    "category": cat.key,
                    "path": f.path,
                    "why": f.why,
                    "headers": f.headers,
                    "content": content_path.read_text(encoding="utf-8"),
                }
            )

    sys_prompt = (
        "Сжато выпиши ключевые заметки для будущего извлечения политик: роли/уровни, чувствительность, "
        "скоупинг, локации, системы, публичные правила, действия/API. "
        "Не выдумывай новые факты. Формат: список коротких bullets."
    )

    client, mdl = _client(model)
    resp = client.beta.chat.completions.parse(
        model=mdl,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": json.dumps(bundle, ensure_ascii=False)[:12000]},
        ],
        response_format=PolicyNotes,
    )

    notes: PolicyNotes = resp.choices[0].message.parsed
    notes.plan_id = notes.plan_id or "manual"
    notes.source_found_path = str(found_path)
    path = FOUND_ROOT / f"policy-notes-{_ts()}.json"
    path.write_text(notes.model_dump_json(indent=2, ensure_ascii=False), encoding="utf-8")
    return path


__all__ = [
    "find_policy",
    "extract_policy_notes",
    "FoundData",
    "PolicyNotes",
]
