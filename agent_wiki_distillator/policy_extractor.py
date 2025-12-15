from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Literal
import argparse

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()

DEFAULT_DOCS_ROOT = Path("sgr-knowledge-agent-erc3_test/docs")
DEFAULT_FOUND_PATH = Path("agent_wiki_distillator/found_data/found-20251215T153124Z/found.json")
FOUND_ROOT = Path("agent_wiki_extraction/found_data")
APIS_DEFAULT_PATH = DEFAULT_DOCS_ROOT / "prep_desc.md"


class FilesCategories(BaseModel):
    file_name: str
    #file_content_category: list [Literal[ "other", "company_overview", "existed_employees", "existed_roles", "existed_systems", "existed_data", "existed_locations", "roles_and_sources_access_level" ] ]
    why: str


#     file_content_category: Literal["employee_access_rules" ("action_access", "data_read_access"), "external_bot_access_rules" ("action_access", "data_read_access"), "internal_bot_access_rules"("action_access", "data_read_access"), "roles_and_sources_access_level", "existed_roles", "existed_systems",  "existed_apis", "existed_data", "existed_locations"] 
# "access_rules",  "people_and_roles", "systems_and_overview", "action_access", "data_read_access", "locations", "apis"] 

class SensitivityLevels(BaseModel):   # security_and_rules
    existed_types_of_sensitivity: list[str]

class RoleTypes(BaseModel):   # people_and_roles
    existed_role_names: list[str] = Field(..., description="CEO, engineer, team lead etc.")
    
class RoleLevels(BaseModel):   # people_and_roles
    existed_levels_of_role_hierarchy: list[str] = Field(..., description="Lvl1, lvl2. etc")
    
class SystemTypes(BaseModel):  # "systems_and_data"
    mentioned_systems_names: list[str] = Field(..., description="CRM, DataBase1, etc.")
 
class DataTypes(BaseModel):    # "systems_and_data" # apis
    mentioned_data_entities: list[str] = Field(..., description="emplyee skills, customer contacts, etc.")

class LocationTypes(BaseModel):  # locations
    existed_location_names: list[str] = Field(..., description="Toronto, New York, etc.")

class ActionTypes(BaseModel):   # security_and_rules # apis
    mentioned_possible_actions: list[str] = Field(..., description="read, update, write, search, etc.")


class Rules(BaseModel):   # security_and_rules
    rules: list[EmployeeAccessRule | ExternalBotAccessRule | InternalBotAccessRule]


class Rule(BaseModel):   # security_and_rules
    type: Literal ["employee_access_rules", "external_bot_access_rules", "internal_bot_access_rules"]
    rule: str
    
class SensitivityRoleMandat(BaseModel): # security_and_rules
    role_level: Literal [RoleLevels.existed_levels_of_role_hierarchy.values]
    max_sensitivity_level_allowed: Literal [SensitivityLevels.existing_types_of_sensitivity.values]
    
class SensitivityDataMandat(BaseModel): # security_and_rules
    data_entity: Literal [DataTypes.mentioned_data_entities.values]
    sensitivity_level: Literal [SensitivityLevels.existing_types_of_sensitivity.values]
    
class SensitivitySystemMandat(BaseModel): # security_and_rules
    system: Literal [SystemTypes.mentioned_systems_names.values]
    sensitivity_level: Literal [SensitivityLevels.existing_types_of_sensitivity.values]
    
class EmployeeAccessRule(BaseModel): # security_and_rules
    type: Literal ["employee_access_rules"]
    role_level: Literal [RoleLevels.existed_levels_of_role_hierarchy.values]
    allowed: list[Transaction]
    deny: list[Transaction]
    
class Transaction(BaseModel): # security_and_rules
    actions: list[Literal[ActionTypes.mentioned_possible_actions.values]]
    data: list[Literal[DataTypes.mentioned_data_entities.values] ]



class ExternalBotAccessRule(BaseModel):
    type: Literal ["external_bot_access_rules"] 
    rule: str 

class InternalBotAccessRule(BaseModel):
    type: Literal ["internal_bot_access_rules"]
    rule: str 


class AccessRules(BaseModel):
    actor: str = ""
    resorce_type: Literal ["api_action", "system_acess", "data_acess"]
    resorce_name: str = ""
    action_allow: str | None
    action_deny: str | None


class SecurityRule(BaseModel):
    path_and_row: str = Field(..., description="Path to file and line number")
    rule: str
    actors: List[str] = Field(default_factory=list, description="People/roles the rule applies to")
    data_scope: List[str] = Field(default_factory=list, description="Data or resources referenced")
    restrictions: List[str] = Field(default_factory=list, description="Allow/deny/conditions")


class SecurityExtraction(CompanyBlock):
    rules: List[SecurityRule] = Field(default_factory=list)


class LocationEntry(BaseModel):
    path: str
    location: str
    address: str | None = None
    contacts: List[str] = Field(default_factory=list)
    notes: str = ""


class LocationsExtraction(CompanyBlock):
    locations: List[LocationEntry] = Field(default_factory=list)


class PersonEntry(BaseModel):
    path: str
    name: str
    role: str
    location: str | None = None
    responsibilities: List[str] = Field(default_factory=list)
    reports_to: str | None = None


class PeopleExtraction(CompanyBlock):
    people: List[PersonEntry] = Field(default_factory=list)


class SystemEntry(BaseModel):
    path: str
    name: str
    description: str
    data_assets: List[str] = Field(default_factory=list)
    sensitivity: str | None = None
    integrations: List[str] = Field(default_factory=list)


class SystemsExtraction(CompanyBlock):
    systems: List[SystemEntry] = Field(default_factory=list)


class ApiEntry(BaseModel):
    path: str
    name: str
    purpose: str
    endpoints: List[str] = Field(default_factory=list)
    auth: str | None = None
    pii_fields: List[str] = Field(default_factory=list)


class ApisExtraction(CompanyBlock):
    apis: List[ApiEntry] = Field(default_factory=list)


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _client(cli: OpenAI | None = None) -> OpenAI:
    if cli:
        return cli
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return OpenAI(api_key=key)


def _load_found(found_path: Path) -> dict:
    if not found_path.exists():
        raise FileNotFoundError(found_path)
    return json.loads(found_path.read_text(encoding="utf-8"))


def _meta(found: dict) -> CompanyBlock:
    return CompanyBlock(
        found_id=found.get("found_id") or "",
        company_name=found.get("company_name") or "",
        company_role=found.get("company_role") or "",
    )


def _read_files(paths: List[str], docs_root: Path) -> List[dict]:
    docs: List[dict] = []
    for p in paths:
        p_obj = Path(p)
        candidates = [p_obj]
        if not p_obj.is_absolute():
            candidates.insert(0, docs_root / p_obj)

        full: Path | None = None
        for cand in candidates:
            if cand.exists():
                full = cand
                break
        if not full:
            continue

        try:
            rel = full.relative_to(docs_root)
            path_label = str(rel)
        except ValueError:
            path_label = str(full)

        docs.append({"path": path_label, "content": full.read_text(encoding="utf-8")})
    return docs


def _save(category: str, payload: BaseModel, root: Path = FOUND_ROOT) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{category}-{_ts()}.json"
    path.write_text(payload.model_dump_json(indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _call_llm(
    category: str,
    schema: type[BaseModel],
    files: List[str],
    company: CompanyBlock,
    docs_root: Path,
    system_hint: str,
    model: str,
    client: OpenAI | None = None,
) -> BaseModel:
    cli = _client(client)
    docs = _read_files(files, docs_root)
    if not docs:
        return schema(**company.model_dump())

    sys_prompt = (
        f"You extract '{category}' facts for {company.company_name} ({company.company_role}). "
        "Use only provided wiki files. Respond strictly in JSON matching the schema."
    )
    if system_hint:
        sys_prompt += f" {system_hint}"

    user_payload = {"company": company.model_dump(), "files": docs}
    resp = cli.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        response_format=schema,
    )
    return resp.choices[0].message.parsed


def extract_security_and_rules(
    files: List[str],
    model: str = "gpt-4.1",
    docs_root: Path = DEFAULT_DOCS_ROOT,
    found_path: Path = DEFAULT_FOUND_PATH,
    client: OpenAI | None = None,
) -> Path:
    found = _load_found(found_path)
    company = _meta(found)
    result: SecurityExtraction = _call_llm(
        "security_and_rules",
        SecurityExtraction,
        files,
        company,
        docs_root,
        system_hint="Capture actors, allowed or denied actions, data scope, and conditions per rule.",
        model=model,
        client=client,
    )
    result.found_id = result.found_id or company.found_id
    result.company_name = result.company_name or company.company_name
    result.company_role = result.company_role or company.company_role
    return _save("security_and_rules", result)


def extract_locations(
    files: List[str],
    model: str = "gpt-4.1",
    docs_root: Path = DEFAULT_DOCS_ROOT,
    found_path: Path = DEFAULT_FOUND_PATH,
    client: OpenAI | None = None,
) -> Path:
    found = _load_found(found_path)
    company = _meta(found)
    result: LocationsExtraction = _call_llm(
        "locations",
        LocationsExtraction,
        files,
        company,
        docs_root,
        system_hint="List offices/locations with address, city/region, and local contacts if stated.",
        model=model,
        client=client,
    )
    result.found_id = result.found_id or company.found_id
    result.company_name = result.company_name or company.company_name
    result.company_role = result.company_role or company.company_role
    return _save("locations", result)


def extract_people_and_roles(
    files: List[str],
    model: str = "gpt-4.1",
    docs_root: Path = DEFAULT_DOCS_ROOT,
    found_path: Path = DEFAULT_FOUND_PATH,
    client: OpenAI | None = None,
) -> Path:
    found = _load_found(found_path)
    company = _meta(found)
    result: PeopleExtraction = _call_llm(
        "people_and_roles",
        PeopleExtraction,
        files,
        company,
        docs_root,
        system_hint="Extract people, titles, reporting lines, and key responsibilities.",
        model=model,
        client=client,
    )
    result.found_id = result.found_id or company.found_id
    result.company_name = result.company_name or company.company_name
    result.company_role = result.company_role or company.company_role
    return _save("people_and_roles", result)


def extract_systems_and_data(
    files: List[str],
    model: str = "gpt-4.1",
    docs_root: Path = DEFAULT_DOCS_ROOT,
    found_path: Path = DEFAULT_FOUND_PATH,
    client: OpenAI | None = None,
) -> Path:
    found = _load_found(found_path)
    company = _meta(found)
    result: SystemsExtraction = _call_llm(
        "systems_and_data",
        SystemsExtraction,
        files,
        company,
        docs_root,
        system_hint="Summarize internal systems, stored data assets, sensitivity markers, and integrations.",
        model=model,
        client=client,
    )
    result.found_id = result.found_id or company.found_id
    result.company_name = result.company_name or company.company_name
    result.company_role = result.company_role or company.company_role
    return _save("systems_and_data", result)


def extract_apis(
    files: List[str],
    model: str = "gpt-4.1",
    docs_root: Path = DEFAULT_DOCS_ROOT,
    found_path: Path = DEFAULT_FOUND_PATH,
    client: OpenAI | None = None,
) -> Path:
    found = _load_found(found_path)
    company = _meta(found)
    result: ApisExtraction = _call_llm(
        "apis",
        ApisExtraction,
        files,
        company,
        docs_root,
        system_hint="Identify APIs/integrations, purpose, typical endpoints or actions, auth, and any PII fields.",
        model=model,
        client=client,
    )
    result.found_id = result.found_id or company.found_id
    result.company_name = result.company_name or company.company_name
    result.company_role = result.company_role or company.company_role
    return _save("apis", result)


def extract_all(
    model: str = "gpt-4.1",
    docs_root: Path = DEFAULT_DOCS_ROOT,
    found_path: Path = DEFAULT_FOUND_PATH,
    client: OpenAI | None = None,
) -> dict[str, Path]:
    found = _load_found(found_path)
    files = found.get("files") or {}
    return {
        "security_and_rules": extract_security_and_rules(files.get("security_and_rules", []), model, docs_root, found_path, client),
        "locations": extract_locations(files.get("locations", []), model, docs_root, found_path, client),
        "people_and_roles": extract_people_and_roles(files.get("people_and_roles", []), model, docs_root, found_path, client),
        "systems_and_data": extract_systems_and_data(files.get("systems_and_data", []), model, docs_root, found_path, client),
        "apis": extract_apis(files.get("apis", []), model, docs_root, found_path, client),
    }


def _cli() -> None:
    """
    Minimal CLI: first arg is optional category, defaults to all.
    Usage:
      python policy_extractor.py              # all categories
      python policy_extractor.py apis         # one category
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("category", nargs="?", default="all")
    parser.add_argument("path", nargs="?", help="Optional path for apis category")
    args, _unknown = parser.parse_known_args()

    category = args.category
    category_map = {
        "security_and_rules": extract_security_and_rules,
        "locations": extract_locations,
        "people_and_roles": extract_people_and_roles,
        "systems_and_data": extract_systems_and_data,
        "apis": extract_apis,
    }

    model = os.environ.get("OPENAI_MODEL", "gpt-4.1")

    if category == "all":
        paths = extract_all(model=model, docs_root=DEFAULT_DOCS_ROOT, found_path=DEFAULT_FOUND_PATH)
        print(json.dumps({k: str(v) for k, v in paths.items()}, ensure_ascii=False, indent=2))
        return

    if category not in category_map:
        print(f"Unknown category: {category}. Use one of: all, {', '.join(category_map)}")
        return

    if category == "apis":
        custom_path = args.path or str(APIS_DEFAULT_PATH)
        files = [custom_path]
    else:
        found = _load_found(DEFAULT_FOUND_PATH)
        files = (found.get("files") or {}).get(category, [])

    path = category_map[category](files=files, model=model, docs_root=DEFAULT_DOCS_ROOT, found_path=DEFAULT_FOUND_PATH)
    print(str(path))


if __name__ == "__main__":
    _cli()
