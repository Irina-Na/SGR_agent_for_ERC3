from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

DEFAULT_INDEX = Path("sgr-knowledge-agent-erc3_test/docs/wiki_index.json")
DEFAULT_DOCS_ROOT = Path("sgr-knowledge-agent-erc3_test/docs")
DEFAULT_FETCH_SCRIPT = Path("sgr-knowledge-agent-erc3_test/fetch_wiki.py")
FOUND_ROOT = Path("agent_wiki_distillator/found_data")

load_dotenv()


class CategoryHits(BaseModel):
    security_rules: List[str] = Field(default_factory=list, description="Policies, rulebooks, access guardrails")
    access_levels: List[str] = Field(default_factory=list, description="Company hierarchy, access levels for roles")
    locations: List[str] = Field(default_factory=list, description="Offices, locations, regional rules")
    people_and_roles: List[str] = Field(default_factory=list, description="Specific employees and their responsibilities")
    systems_and_data: List[str] = Field(default_factory=list, description="Internal systems, data sets, storage")
    apis: List[str] = Field(default_factory=list, description="Available or documented APIs")


class WikiDiscovery(BaseModel):
    company_name: str | None = Field(default=None, description="Company name if stated in README")
    company_role: str | None = Field(default=None, description="Company role/industry if stated in README")
    files: CategoryHits
    what_can_be_find_in_wiki: str
    what_information_is_missing_from_the_wiki: str


def _unique_id(prefix: str = "found") -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{ts}"


def _ensure_index(index_path: Path = DEFAULT_INDEX, fetch_script: Path = DEFAULT_FETCH_SCRIPT) -> dict:
    if index_path.exists():
        return json.loads(index_path.read_text(encoding="utf-8"))
    if fetch_script.exists():
        subprocess.run([sys.executable, str(fetch_script)], check=True)
    if not index_path.exists():
        raise FileNotFoundError(f"wiki index missing: {index_path}")
    return json.loads(index_path.read_text(encoding="utf-8"))


def _save_found(run_id: str, payload: WikiDiscovery, root: Path = FOUND_ROOT) -> Path:
    path_root = root / run_id
    path_root.mkdir(parents=True, exist_ok=True)
    path = path_root / "found.json"
    data = {"found_id": run_id, **payload.model_dump()}
    path.write_text(json.dumps(data, ensure_ascii=True, indent=2), encoding="utf-8")
    return path


def find_categories(
    model: str = "gpt-5.1",
    index_path: Path = DEFAULT_INDEX,
    docs_root: Path = DEFAULT_DOCS_ROOT,
    fetch_script: Path = DEFAULT_FETCH_SCRIPT,
    client: OpenAI | None = None,
) -> tuple[WikiDiscovery, Path]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    cli = client or OpenAI(api_key=api_key)

    wiki_index = _ensure_index(index_path=index_path, fetch_script=fetch_script)
    files = wiki_index.get("files", [])

    readme_path = docs_root / "README.md"
    readme_content = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

    entries = []
    for f in files:
        rel = f.get("path")
        if not rel:
            continue
        headers = f.get("headers") or []
        entries.append(
            {
                "path": rel,
                "tokens": f.get("tokens"),
                "headers": headers,
            }
        )

    system_prompt = ( """You are an analyst-consultant helping to find files within the company’s wiki documentation that contain the target data category for further study.
You are given the documentation of a company specified in the README file. Extract company_name and company_role from the README.
Look at the list of wiki files and their titles. Determine what information can be found inside each of the files by category - select the data categories that can be found in the file.
Use only paths that are present in the index. Do not invent paths. The answer must be strictly in JSON according to the schema.""")
    user_prompt = (f"README content:\n{readme_content}"
        "Wiki index entries (path, tokens, headers):\n"
        f"{json.dumps(entries, ensure_ascii=False, indent=2)}\n\n"
    )

    resp = cli.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format=WikiDiscovery,
    )

    found: WikiDiscovery = resp.choices[0].message.parsed
    run_id = _unique_id("found")
    out_path = _save_found(run_id, found, root=FOUND_ROOT)
    return found, out_path


if __name__ == "__main__":
    try:
        model = os.environ.get("OPENAI_MODEL", "gpt-4.1")
        found, path = find_categories(model=model)
        print(f"saved {path}")
    except Exception as e:  # noqa: BLE001
        print(f"error: {e}", file=sys.stderr)
        raise
