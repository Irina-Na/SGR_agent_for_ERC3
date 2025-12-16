from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from agent_wiki_distillator.data_models import (
    CompanyBlock,
    Rules,
    SystemApiCoverageExtraction,
)

load_dotenv()

DEFAULT_DOCS_ROOT = Path("sgr-knowledge-agent-erc3_test/docs")
DEFAULT_FOUND_ROOT = Path("agent_wiki_distillator/found_data")
DEFAULT_OUTPUT_DIR = Path("sgr-knowledge-agent-erc3_test/extracted_data")


def _client(cli: OpenAI | None = None) -> OpenAI:
    if cli:
        return cli
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return OpenAI(api_key=key)


def _resolve_found(found_path: Path | None) -> Path:
    if found_path and Path(found_path).exists():
        return Path(found_path)
    candidates = sorted(DEFAULT_FOUND_ROOT.glob("found-*/found.json"), reverse=True)
    if candidates:
        return candidates[0]
    raise FileNotFoundError("found.json not found; run wiki_annotator first.")


def _read_found(found_path: Path | None) -> dict:
    resolved = _resolve_found(found_path)
    return json.loads(resolved.read_text(encoding="utf-8"))


def _read_docs(paths: List[str], docs_root: Path) -> List[dict]:
    docs: List[dict] = []
    for p in paths:
        path = docs_root / p if not Path(p).is_absolute() else Path(p)
        if not path.exists():
            continue
        try:
            rel = path.relative_to(docs_root)
            label = str(rel)
        except ValueError:
            label = str(path)
        docs.append({"path": label, "content": path.read_text(encoding="utf-8")})
    return docs


def _call_llm(
    category: str,
    schema: type[BaseModel],
    files: List[str],
    docs_root: Path,
    company_name: str,
    company_role: str,
    system_hint: str,
    model: str,
    client: OpenAI | None = None,
    base_payload: dict | None = None,
) -> BaseModel:
    docs = _read_docs(files, docs_root)
    if not docs:
        return schema(**(base_payload or {}))  # type: ignore[arg-type]

    sys_prompt = (
        f"Extract '{category}' for {company_name} ({company_role}). "
        "Use only provided wiki content. Reply strictly as JSON schema."
    )
    if system_hint:
        sys_prompt += f" {system_hint}"

    payload = {"company_name": company_name, "company_role": company_role, "files": docs}
    resp = _client(client).beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        response_format=schema,
    )
    return resp.choices[0].message.parsed


def distill_bundle(
    found_path: Path | None = None,
    docs_root: Path = DEFAULT_DOCS_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    model: str | None = None,
    client: OpenAI | None = None,
) -> Path:
    found = _read_found(found_path)
    files = found.get("files") or {}
    company_name = found.get("company_name", "")
    company_role = found.get("company_role", "")
    model_id = model or os.environ.get("OPENAI_MODEL", "gpt-4.1")

    base = {
        "found_id": found.get("found_id", ""),
        "company_name": company_name,
        "company_role": company_role,
        "company_locations": [],
    }

    company_files: List[str] = []
    if "locations" in files:
        company_files.extend(files["locations"])
    readme = docs_root / "README.md"
    if readme.exists():
        company_files.append(str(readme))
    company = _call_llm(
        "company_profile",
        CompanyBlock,
        company_files,
        docs_root,
        company_name,
        company_role,
        "Return company_name, company_role, and company_locations (list of objects with company_location, specification).",
        model_id,
        client,
        base_payload=base,
    )

    rules_files = list(
        dict.fromkeys(
            (files.get("security_rules") or [])
            + (files.get("access_levels") or [])
            + (files.get("security_rules_and_access_levels") or [])
            + (files.get("security_and_rules") or [])
        )
    )
    rules = _call_llm(
        "security_rules_and_access_levels",
        Rules,
        rules_files,
        docs_root,
        company_name,
        company_role,
        "Capture actors, allowed/denied actions, scope, and conditions per rule. Each SecurityRule must include category.",
        model_id,
        client,
        base_payload=base,
    )

    coverage_files = list(dict.fromkeys((files.get("systems_and_data") or []) + (files.get("apis") or [])))
    coverage = _call_llm(
        "system_api_coverage",
        SystemApiCoverageExtraction,
        coverage_files,
        docs_root,
        company_name,
        company_role,
        "Match systems with APIs; if no API set has_api=false with missing_reason.",
        model_id,
        client,
        base_payload=base,
    )

    bundle = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "found_path": str(_resolve_found(found_path)),
        "company": company.model_dump(),
        "rules": rules.model_dump(),
        "system_api_coverage": coverage.model_dump(),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"wiki_policy_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    out_path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def _cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("category", nargs="?", default="bundle")
    parser.add_argument("found", nargs="?", help="Path to found.json")
    args = parser.parse_args()
    if args.category != "bundle":
        print("Only bundle mode is supported; run without args.")
        return
    path = distill_bundle(found_path=Path(args.found) if args.found else None)
    print(path)


if __name__ == "__main__":
    _cli()
