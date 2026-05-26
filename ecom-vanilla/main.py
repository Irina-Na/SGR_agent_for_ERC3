import os
import sys
import textwrap
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path

from bitgn.harness_connect import HarnessServiceClientSync
from bitgn.harness_pb2 import (
    EndTrialRequest,
    EvalPolicy,
    GetBenchmarkRequest,
    StartRunRequest,
    StartTrialRequest,
    StatusRequest,
    SubmitRunRequest,
)
from connectrpc.errors import ConnectError

from agent import run_agent


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


def main(run_stem: str = "ecom", trace_dir: Path | None = None) -> None:
    task_filter = os.sys.argv[1:]
    scores = []

    try:
        client = HarnessServiceClientSync(BITGN_URL)
        print("Connecting to BitGN", client.status(StatusRequest()))
        res = client.get_benchmark(GetBenchmarkRequest(benchmark_id=BENCH_ID))
        print(
            f"{EvalPolicy.Name(res.policy)} benchmark: {res.benchmark_id} "
            f"with {len(res.tasks)} tasks.\n{CLI_GREEN}{res.description}{CLI_CLR}"
        )

        run = client.start_run(
            StartRunRequest(
                name="ECOM Python Sample",
                benchmark_id=BENCH_ID,
                api_key=BITGN_API_KEY,
            )
        )

        try:
            for trial_id in run.trial_ids:
                trial = client.start_trial(
                    StartTrialRequest(trial_id=trial_id),
                )
                if task_filter and trial.task_id not in task_filter:
                    continue

                print(f"{'=' * 30} Starting task: {trial.task_id} {'=' * 30}")
                print(f"{CLI_BLUE}{trial.instruction}{CLI_CLR}\n{'-' * 80}")
                try:
                    run_agent(
                        MODEL_ID,
                        trial.harness_url,
                        trial.instruction,
                        provider=PROVIDER,
                        trace_dir=trace_dir,
                        trace_prefix=f"{run_stem}_{trial.task_id}",
                    )
                except Exception as exc:
                    print(exc)

                result = client.end_trial(EndTrialRequest(trial_id=trial.trial_id))
                if result.score_available:
                    scores.append((trial.task_id, result.score))
                    style = CLI_GREEN if result.score == 1 else CLI_RED
                    explain = textwrap.indent("\n".join(result.score_detail), "  ")
                    print(
                        f"\n{style}Score: {result.score:0.2f}\n{explain}\n{CLI_CLR}"
                    )
                else:
                    print(f"\n{CLI_BLUE}Score: not available{CLI_CLR}\n")
        finally:
            client.submit_run(SubmitRunRequest(run_id=run.run_id, force=True))

    except ConnectError as exc:
        print(f"{exc.code}: {exc.message}")
    except KeyboardInterrupt:
        print(f"{CLI_RED}Interrupted{CLI_CLR}")

    if scores:
        for task_id, score in scores:
            style = CLI_GREEN if score == 1 else CLI_RED
            print(f"{task_id}: {style}{score:0.2f}{CLI_CLR}")

        total = sum(score for _, score in scores) / len(scores) * 100.0
        print(f"FINAL: {total:0.2f}%")


if __name__ == "__main__":
    runs_dir = Path(__file__).resolve().parent / "runs"
    runs_dir.mkdir(exist_ok=True)
    log_path = runs_dir / f"ecom_{datetime.now():%Y%m%d_%H%M%S}.log"
    trace_dir = runs_dir / "traces"
    trace_dir.mkdir(exist_ok=True)

    with log_path.open("w", encoding="utf-8") as log_file:
        with redirect_stdout(Tee(sys.stdout, log_file)), redirect_stderr(
            Tee(sys.stderr, log_file)
        ):
            print(f"Logging to {log_path}")
            print(f"LLM traces to {trace_dir}\\{log_path.stem}_<task>_step_<n>.json")
            main(run_stem=log_path.stem, trace_dir=trace_dir)
