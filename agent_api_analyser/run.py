import argparse
import os
from pathlib import Path

from erc3 import ERC3
from dotenv import load_dotenv

from agent_api_analyser.catalog_builder import CatalogBuilder
from agent_api_analyser.executor import Executor
from agent_api_analyser.evaluator import Evaluator
from agent_api_analyser.fixture_resolver import FixtureResolver
from agent_api_analyser.report_builder import ReportBuilder
from agent_api_analyser.scenario_generator import ScenarioGenerator
from agent_api_analyser.wrapper_suggester import WrapperSuggester
from agent_api_analyser.scenario_logger import write_scenarios


def _start_api():
    load_dotenv()  # ensure ERC3_API_KEY from root .env
    core = ERC3()
    session = core.start_session(
        benchmark=os.getenv("ERC3_BENCHMARK", "erc3-dev"),
        workspace=os.getenv("ERC3_WORKSPACE", "api-research"),
        name="api-analyser",
        architecture="api-analyser",
    )
    status = core.session_status(session.session_id)
    if not status.tasks:
        raise RuntimeError("No tasks returned in session")
    task = status.tasks[0]
    core.start_task(task)
    return core.get_erc_client(task), task


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--allow-writes", action="store_true")
    parser.add_argument("--docs", default="sgr-knowledge-agent-erc3_test/docs")
    args = parser.parse_args()

    api, _task = _start_api()

    catalog = CatalogBuilder().build()
    fixtures = FixtureResolver()
    fixtures.prime(api)

    generator = ScenarioGenerator(args.model)
    docs_text = generator.load_docs(args.docs)
    scenarios = generator.generate(catalog, docs_text, "read")
    scenarios_path = write_scenarios(scenarios)

    results = Executor(api, fixtures).run(scenarios, allow_writes=args.allow_writes)
    outcome = Evaluator().evaluate(results)
    suggestions = WrapperSuggester().suggest(results)

    path = ReportBuilder().write(outcome, suggestions)
    print(f"Scenarios saved to {Path(scenarios_path).as_posix()}")
    print(f"Report saved to {Path(path).as_posix()}")


if __name__ == "__main__":
    main()
