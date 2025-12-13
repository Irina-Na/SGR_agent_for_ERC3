from erc3 import ApiException


class FixtureResolver:
    def __init__(self) -> None:
        self.values: dict[str, str | int | None] = {}
        # normalize common aliases coming from LLMs to API-required keys
        self.alias_map: dict[str, str] = {
            "customer_id": "id",
            "employee_id": "id",
            "project_id": "id",
            "wiki_id": "file",
        }

    def _first_id(self, fn, attr: str) -> str | None:
        try:
            res = fn(offset=0, limit=1)
        except ApiException:
            return None
        items = getattr(res, attr, None) or []
        item = items[0] if items else None
        return getattr(item, "id", None)

    def prime(self, api) -> None:
        self.values["employee_id"] = self._first_id(api.search_employees, "employees")
        self.values["project_id"] = self._first_id(api.search_projects, "projects")
        self.values["customer_id"] = self._first_id(api.search_customers, "companies")
        self.values["time_entry_id"] = self._first_id(api.search_time_entries, "entries")

    def fill(self, args: dict) -> dict:
        if not args:
            return {}
        out = {}
        for k, v in args.items():
            key = self.alias_map.get(k, k)
            if isinstance(v, str) and v.startswith("$"):
                out[key] = self.values.get(v[1:], v)
            else:
                out[key] = v
        return out
