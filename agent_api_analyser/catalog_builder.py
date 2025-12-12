from typing import Literal

from erc3 import erc3 as dev
from pydantic import BaseModel


class CatalogEntry(BaseModel):
    name: str
    kind: Literal["read", "write"]
    required: list[str]


class CatalogBuilder:
    def __init__(self, module=dev) -> None:
        self.module = module

    def _guess_kind(self, name: str) -> Literal["read", "write"]:
        writes = ("Update", "Log", "Create", "Delete", "Provide")
        return "write" if any(w in name for w in writes) else "read"

    def build(self) -> list[CatalogEntry]:
        out: list[CatalogEntry] = []
        for name in dir(self.module):
            obj = getattr(self.module, name)
            if (
                isinstance(obj, type)
                and issubclass(obj, BaseModel)
                and name.startswith("Req_")
            ):
                schema = obj.model_json_schema()
                out.append(
                    CatalogEntry(
                        name=name,
                        kind=self._guess_kind(name),
                        required=schema.get("required", []),
                    )
                )
        return out
