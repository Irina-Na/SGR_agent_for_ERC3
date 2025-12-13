import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

from agent_api_analyser.scenario_generator import Scenario


def write_scenarios(scenarios: Iterable[Scenario], out_dir: str | Path = "agent_api_analyser/api-report/scenarios") -> Path:
    """Persist generated scenarios to a timestamped JSON file."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    target = out_path / f"scenarios_{timestamp}.json"
    payload = [s.model_dump(mode="json") for s in scenarios]
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    return target
