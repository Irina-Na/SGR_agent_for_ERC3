"""
CLI entrypoints for security policy planning/extraction.

Usage examples:
  # end-to-end: plan + extract
  python -m agent_security_analyser.cli materialize

  # plan only (inspect/edit plan.json yourself)
  python -m agent_security_analyser.cli plan

  # extract using an existing plan id
  python -m agent_security_analyser.cli extract --plan-id <id>
"""
from __future__ import annotations

import argparse
from pathlib import Path

from agent_security_analyser.policy_planner import (
    DEFAULT_DOCS_ROOT,
    PLANS_ROOT,
    POLICY_ROOT,
    SecurityPolicyPlanner,
    ensure_index,
    load_plan,
    materialize_plan_and_policies,
    save_plan,
    save_policies,
)


def cmd_materialize(args: argparse.Namespace) -> None:
    plan_path, policy_path = materialize_plan_and_policies(
        max_batch_tokens=args.max_batch_tokens,
        model=args.model,
    )
    print(f"Plan saved to   : {plan_path}")
    print(f"Policies saved to: {policy_path}")


def cmd_plan(args: argparse.Namespace) -> None:
    idx = ensure_index()
    planner = SecurityPolicyPlanner(model=args.model)
    plan_id, plan = planner.generate_plan(idx, max_batch_tokens=args.max_batch_tokens)
    plan_path = save_plan(plan_id, plan, plans_root=PLANS_ROOT)
    print(f"Plan {plan_id} saved to: {plan_path}")


def cmd_extract(args: argparse.Namespace) -> None:
    idx = ensure_index()
    plan_id, plan = load_plan(Path(args.plan_path))
    planner = SecurityPolicyPlanner(model=args.model)
    policies = planner.run_plan(
        plan,
        docs_root=DEFAULT_DOCS_ROOT,
        wiki_index=idx,
        max_batch_tokens=args.max_batch_tokens,
    )
    policy_path = save_policies(plan_id, policies, policy_root=POLICY_ROOT)
    print(f"Policies saved to: {policy_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Security policy planner CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_mat = sub.add_parser("materialize", help="plan + extract policies")
    p_mat.add_argument("--max-batch-tokens", type=int, default=6000)
    p_mat.add_argument("--model", type=str, default="gpt-4o-mini")
    p_mat.set_defaults(func=cmd_materialize)

    p_plan = sub.add_parser("plan", help="generate plan only")
    p_plan.add_argument("--max-batch-tokens", type=int, default=6000)
    p_plan.add_argument("--model", type=str, default="gpt-4o-mini")
    p_plan.set_defaults(func=cmd_plan)

    p_ext = sub.add_parser("extract", help="extract policies from existing plan.json")
    p_ext.add_argument("--plan-path", type=str, required=True, help="Path to plan.json")
    p_ext.add_argument("--max-batch-tokens", type=int, default=6000)
    p_ext.add_argument("--model", type=str, default="gpt-4o-mini")
    p_ext.set_defaults(func=cmd_extract)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
