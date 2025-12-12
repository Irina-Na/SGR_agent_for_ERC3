from pydantic import BaseModel

from agent_api_analyser.executor import ExecutionResult


class EvalOutcome(BaseModel):
    results: list[ExecutionResult]
    passed: int
    failed: int
    skipped: int


class Evaluator:
    def evaluate(self, results: list[ExecutionResult]) -> EvalOutcome:
        passed = sum(1 for r in results if r.ok)
        skipped = sum(1 for r in results if r.error and r.error.startswith("skipped"))
        failed = len(results) - passed - skipped
        return EvalOutcome(results=results, passed=passed, failed=failed, skipped=skipped)

