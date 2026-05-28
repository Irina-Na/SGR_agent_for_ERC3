"""
BitGN harness runner. Discovery is hoisted to run-scope: it executes once
on the first matched trial's VM and is reused for every subsequent trial
in the run.
"""
from __future__ import annotations

import os
import sys
import textwrap
from typing import Optional

from connectrpc.errors import ConnectError

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
PROVIDER = os.getenv("PROVIDER") or "openai"
MODEL_ID = os.getenv("MODEL_ID") or (
    "openai/gpt-oss-120b" if PROVIDER == "nebius" else "gpt-4.1-2025-04-14"
)

CLI_RED = "\x1B[31m"
CLI_GREEN = "\x1B[32m"
CLI_BLUE = "\x1B[34m"
CLI_CLR = "\x1B[0m"


def main() -> None:
    task_filter = sys.argv[1:]
    scores: list[tuple[str, float]] = []
    discovery: Optional[SessionDiscovery] = None

    try:
        client = HarnessServiceClientSync(BITGN_URL)
        print("Connecting to BitGN", client.status(StatusRequest()))
        bench = client.get_benchmark(GetBenchmarkRequest(benchmark_id=BENCH_ID))
        print(
            f"{EvalPolicy.Name(bench.policy)} benchmark: {bench.benchmark_id} "
            f"with {len(bench.tasks)} tasks.\n{CLI_GREEN}{bench.description}{CLI_CLR}"
        )

        run = client.start_run(StartRunRequest(
            name=f"ECOM Knowledge Agent v0.1 ({MODEL_ID})",
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
    main()
