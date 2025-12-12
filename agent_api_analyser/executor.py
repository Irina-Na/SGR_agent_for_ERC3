from erc3 import ApiException, erc3 as dev
from pydantic import BaseModel

from agent_api_analyser.fixture_resolver import FixtureResolver
from agent_api_analyser.scenario_generator import Scenario


class ExecutionResult(BaseModel):
    scenario: Scenario
    ok: bool
    error: str | None = None
    payload: str | None = None


class Executor:
    def __init__(self, api, fixtures: FixtureResolver) -> None:
        self.api = api
        self.fixtures = fixtures

    @staticmethod
    def _args_to_dict(scenario: Scenario) -> dict:
        if not scenario.args:
            return {}
        if isinstance(scenario.args, list):
            return {arg.name: arg.value for arg in scenario.args}
        return scenario.args

    def _build_req(self, scenario: Scenario):
        cls = getattr(dev, scenario.request, None)
        if not cls:
            raise ValueError(f"Unknown request: {scenario.request}")
        args = self._args_to_dict(scenario)
        return cls(**self.fixtures.fill(args))

    def run(self, scenarios: list[Scenario], allow_writes: bool = False) -> list[ExecutionResult]:
        results: list[ExecutionResult] = []
        for sc in scenarios:
            if sc.kind == "write" and not allow_writes:
                results.append(ExecutionResult(scenario=sc, ok=False, error="skipped: write blocked"))
                continue
            try:
                req = self._build_req(sc)
                resp = self.api.dispatch(req)
                payload = resp.model_dump_json(exclude_none=True, exclude_unset=True)
                results.append(ExecutionResult(scenario=sc, ok=True, payload=payload))
            except ApiException as exc:
                results.append(ExecutionResult(scenario=sc, ok=False, error=exc.detail))
            except Exception as exc:  # noqa: BLE001
                results.append(ExecutionResult(scenario=sc, ok=False, error=str(exc)))
        return results
