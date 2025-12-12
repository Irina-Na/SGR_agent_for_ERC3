import os
from pathlib import Path
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field


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
        root = Path(path)
        return "\n\n".join(p.read_text(encoding="utf-8") for p in sorted(root.rglob("*.md")))

    def generate(
        self, catalog: list, docs_text: str, kind: Literal["read", "write"] = "read"
    ) -> list[Scenario]:
        allowed = [c.name for c in catalog if getattr(c, "kind", "read") == kind]
        prompt = f"""You are creating {kind} API checks.
Use only request names from: {allowed}
Use placeholders $employee_id, $project_id, $customer_id, $time_entry_id.
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
