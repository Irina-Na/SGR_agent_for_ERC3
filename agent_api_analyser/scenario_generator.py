import os
from pathlib import Path
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field
from agent_api_analyser.wiki_context import build_wiki_context


class Argument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    value: str | int | float | bool | None = None


class Scenario(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    request: str
    args: list[Argument] = Field(default_factory=list)
    kind: Literal["read", "write"] = "read"
    note: str | None = None


class _ScenarioList(BaseModel):
    scenarios: list[Scenario]


class ScenarioGenerator:
    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = model

    @staticmethod
    def load_docs(path: str | Path = "sgr-knowledge-agent-erc3_test/docs") -> str:
        # Prefer curated wiki context if index is present, otherwise fallback to raw docs concat.
        return build_wiki_context(path)

    def generate(
        self, catalog: list, docs_text: str, kind: Literal["read", "write"] = "read"
    ) -> list[Scenario]:
        allowed = [c for c in catalog if getattr(c, "kind", "read") == kind]
        allowed_names = [c.name for c in allowed]
        required_by_req = {
            c.name: (c.required or []) for c in allowed
        }
        field_specs = {c.name: c.fields for c in allowed}
        prompt = f"""You are creating {kind} API checks.
Use only request names from: {allowed_names}
Each scenario must include all required fields for the chosen request exactly by name:
{required_by_req}
- Field specs (types/defaults/limits/enums/format) per request:
{field_specs}
- Respect defaults when present; otherwise pick safe values within min/max or from enum.
- Always include pagination args when required: offset=0, limit=20
- When date_from/date_to are required, provide an ISO range like "2025-01-01" and "2025-01-31"
- Use placeholders $employee_id, $project_id, $customer_id, $time_entry_id only for ids;
  never rename required keys (use 'id' not 'customer_id'/'employee_id', and 'file' for wiki files).
Return at most 8 scenarios."""
        messages = [
            {"role": "system", "content": "Generate lean API test intents for an internal agent."},
            {"role": "user", "content": prompt + "\n\nContext:\n" + docs_text[:6000]},
        ]
        resp = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=messages,
            response_format=_ScenarioList,
        )
        return resp.choices[0].message.parsed.scenarios
