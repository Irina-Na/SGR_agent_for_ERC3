"""
BitGN harness runner. Mirrors ecom-vanilla's logging (Tee to runs/<...>.log +
per-step trace dumps). Discovery is hoisted to run-scope: it executes once on the
first matched trial's VM and is reused for every subsequent trial in the run.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import textwrap
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import Optional

from connectrpc.errors import ConnectError

from bitgn.harness_connect import HarnessServiceClientSync
from bitgn.harness_pb2 import (
    EndTrialRequest,
    EvalPolicy,
    GetBenchmarkRequest,
    GetRunRequest,
    StartRunRequest,
    StartTrialRequest,
    StatusRequest,
    SubmitRunRequest,
    TRIAL_STATE_DONE,
)
from bitgn.vm.ecom.ecom_connect import EcomRuntimeClientSync

from agent import get_llm_client, run_agent
from ecom_discovery import SessionDiscovery, discover


BITGN_URL = (
    os.getenv("BITGN_HOST")
    or os.getenv("BENCHMARK_HOST")
    or "https://api.bitgn.com"
)
BITGN_API_KEY = os.getenv("BITGN_API_KEY") or ""
BENCH_ID = os.getenv("BENCH_ID") or os.getenv("BENCHMARK_ID") or "bitgn/ecom1-dev"
PROVIDER = os.getenv("PROVIDER") or "nebius"  # "nebius" or "openai"
MODEL_ID = os.getenv("MODEL_ID") or (
    "openai/gpt-oss-120b" if PROVIDER == "nebius" else "gpt-4.1-2025-04-14"
)

_VERSION = "0.4.2"
try:
    with Path(__file__).with_name("pyproject.toml").open("rb") as _f:
        import tomllib
        _VERSION = tomllib.load(_f).get("project", {}).get("version", _VERSION_DEFAULT)
except Exception:
    pass
VERSION = _VERSION

CLI_RED = "\x1B[31m"
CLI_GREEN = "\x1B[32m"
CLI_CLR = "\x1B[0m"
CLI_BLUE = "\x1B[34m"


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def _safe_filename_part(value: str) -> str:
    value = value.strip()
    value = re.sub(r"[^\w.-]+", "_", value)
    value = value.strip("._")
    return value or "unknown"


def _last_commit_id() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(1, 1000):
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"Could not find a free report filename for {path}")


def main(run_stem: str = "ecom", trace_dir: Path | None = None) -> float | None:
    task_filter = os.sys.argv[1:]
    scores: list[tuple[str, float]] = []
    final_score: float | None = None
    discovery: Optional[SessionDiscovery] = None

    try:
        client = HarnessServiceClientSync(BITGN_URL)
        print("Connecting to BitGN", client.status(StatusRequest()))
        res = client.get_benchmark(GetBenchmarkRequest(benchmark_id=BENCH_ID))
        print(
            f"{EvalPolicy.Name(res.policy)} benchmark: {res.benchmark_id} "
            f"with {len(res.tasks)} tasks.\n{CLI_GREEN}{res.description}{CLI_CLR}"
        )

        run = client.start_run(StartRunRequest(
            name=f"@Irinai_Na Knowledge Agent v{VERSION} ({MODEL_ID})",
            benchmark_id=BENCH_ID,
            api_key=BITGN_API_KEY,
        ))

        try:
            for trial_id in run.trial_ids:
                trial = client.start_trial(StartTrialRequest(trial_id=trial_id))
                if task_filter and trial.task_id not in task_filter:
                    continue

                # Run-scoped discovery: build once on the first matched trial's VM.
                if discovery is None:
                    print(f"{CLI_BLUE}=== Bootstrapping discovery on {trial.task_id} ==={CLI_CLR}")
                    vm = EcomRuntimeClientSync(trial.harness_url)
                    llm = get_llm_client(PROVIDER)
                    discovery = discover(vm, llm, MODEL_ID)
                    print(
                        f"{CLI_BLUE}Discovery: "
                        f"sql_tool={discovery.sql_tool}, "
                        f"identity_tool={discovery.identity_tool}, "
                        f"time_tool={discovery.time_tool}, "
                        f"entity_kinds={discovery.entity_kinds}, "
                        f"docs={len(discovery.docs_tree)}, "
                        f"tools={len(discovery.tool_index)}{CLI_CLR}"
                    )

                print(f"{'=' * 30} Starting task: {trial.task_id} {'=' * 30}")
                print(f"{CLI_BLUE}{trial.instruction}{CLI_CLR}\n{'-' * 80}")
                try:
                    run_agent(
                        MODEL_ID,
                        trial.harness_url,
                        trial.instruction,
                        provider=PROVIDER,
                        discovery=discovery,
                        trace_dir=trace_dir,
                        trace_prefix=f"{run_stem}_{trial.task_id}",
                    )
                except Exception as exc:
                    print(exc)

                client.end_trial(EndTrialRequest(trial_id=trial.trial_id))
                print(f"\n{CLI_BLUE}Trial closed; score will be printed after run submit{CLI_CLR}\n")
        finally:
            print(f"\n{CLI_GREEN}>>>> Submitting run... <<<<{CLI_CLR}")
            client.submit_run(SubmitRunRequest(run_id=run.run_id, force=True))
            result = client.get_run(GetRunRequest(run_id=run.run_id))

            if getattr(result, "score_available", False):
                print(f"FINAL SCORE: {result.score:0.2f}")
                final_score = result.score * 100.0
                incomplete = 0
                for t in result.trials:
                    if t.state != TRIAL_STATE_DONE:
                        incomplete += 1
                        continue

                    style = CLI_GREEN if t.score == 1 else CLI_RED
                    detail = getattr(t, "score_detail", ())
                    explain = "\n" + textwrap.indent("\n".join(detail), "  ") + "\n" if detail else ""
                    print(f"- {t.task_id}: {style}Score: {t.score:0.2f}{CLI_CLR}{explain}".strip("\n "))
                    scores.append((t.task_id, t.score))

                if incomplete > 0:
                    print(f"{CLI_RED}incomplete trials: {incomplete}{CLI_CLR}")
            else:
                print(f"\n{CLI_RED}Score is not available. Results are sealed and will be revealed later{CLI_CLR}\n")

    except ConnectError as exc:
        print(f"{exc.code}: {exc.message}")
    except KeyboardInterrupt:
        print(f"{CLI_RED}Interrupted{CLI_CLR}")

    if final_score is not None:
        return final_score

    if scores:
        return sum(score for _, score in scores) / len(scores) * 100.0

    return None


if __name__ == "__main__":
    runs_dir = Path(__file__).resolve().parent / "runs"
    runs_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_part = _safe_filename_part(MODEL_ID)
    commit_part = _safe_filename_part(_last_commit_id())
    log_path = runs_dir / f"{timestamp}_{model_part}_git{commit_part}.log"
    trace_dir = runs_dir / "traces"
    trace_dir.mkdir(exist_ok=True)
    final_score = None

    with log_path.open("w", encoding="utf-8") as log_file:
        with redirect_stdout(Tee(sys.stdout, log_file)), redirect_stderr(
            Tee(sys.stderr, log_file)
        ):
            print(f"Logging to {log_path}")
            print(f"LLM traces to {trace_dir}\\{log_path.stem}_<task>_step_<n>.json")
            final_score = main(run_stem=log_path.stem, trace_dir=trace_dir)

    score_part = "score_na" if final_score is None else f"score_{final_score:0.2f}"
    final_log_path = _unique_path(
        runs_dir / f"{timestamp}_{model_part}_{score_part}_git{commit_part}.log"
    )
    if final_log_path != log_path:
        log_path.replace(final_log_path)
        print(f"Final report: {final_log_path}")
