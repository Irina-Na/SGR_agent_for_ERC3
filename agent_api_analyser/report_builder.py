from datetime import datetime
from pathlib import Path

from agent_api_analyser.evaluator import EvalOutcome


class ReportBuilder:
    def write(
        self,
        outcome: EvalOutcome,
        suggestions: list[str],
        out_dir: str | Path = "agent_api_analyser/api-report",
    ) -> Path:
        now = datetime.now()
        date_str = now.date().isoformat()
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        target = out_path / f"{timestamp}.md"

        lines: list[str] = []
        lines.append(f"# API check {date_str}")
        lines.append(f"- passed: {outcome.passed}")
        lines.append(f"- failed: {outcome.failed}")
        lines.append(f"- skipped: {outcome.skipped}")
        lines.append("")
        lines.append("## Scenarios")
        for r in outcome.results:
            status = "OK" if r.ok else "ERR"
            lines.append(f"- {status} {r.scenario.request} :: {r.scenario.title}")
            if r.error:
                lines.append(f"  - error: {r.error}")
            if r.payload:
                lines.append(f"  - payload: {r.payload[:500]}")
        if suggestions:
            lines.append("")
            lines.append("## Wrapper ideas")
            for s in suggestions:
                lines.append(f"- {s}")

        target.write_text("\n".join(lines), encoding="utf-8")
        return target
