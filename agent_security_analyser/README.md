Security Checker
================

Purpose
-------
Runtime security gate for ERC3 agents. Given a free-form user request (`user_query`), user context, resource context, and a policy bundle, it returns one of `allow | deny | clarify` with a short reason.

Current layout
--------------
- `agent_security_analyser/security_checker.py`: core LLM-backed classification logic.
- `agent_security_analyser/debug_failed_tasks.py`: helper to replay/classify known bad cases and log LLM decisions.
- `agent_security_analyser/policy/plan-ideal-manual/policies.json`: hand-curated policy bundle shipped with the repo (default for LLM mode).

Prereqs
-------
- `OPENAI_API_KEY` set (or present in repo-root `.env`, automatically loaded by `debug_failed_tasks.py`; set manually when using `security_checker.llm_classify` elsewhere).
- Python deps: `openai`, `pydantic` (optional but required for LLM mode).

How security_checker works
--------------------------
`security_checker.py` exposes:
- `Decision`: dataclass with `status: allow|deny|clarify` and `reason` code.
- `load_policy_constraints(path)`: loads `constraints_map` from a policies file (dict keyed by `user_query`).
- `classify(...)`: legacy alias to `llm_classify`; all decisions go through the LLM (rule-based path removed).
- `llm_classify(user_query, user_ctx, resource_ctx, policy_path=DEFAULT_POLICY_PATH, model=env OPENAI_MODEL or gpt-4o-mini)`: LLM-backed classifier.
  - Builds a strict system prompt with the matching constraint and relevant rules from `policies.json` (or a passed policy dict) and asks the model to return JSON matching `LlmDecision` schema.
  - On parse failure, returns `Decision("deny", "llm_parse_error")`.

Data inputs
-----------
- Default manual policy bundle: `agent_security_analyser/policy/plan-ideal-manual/policies.json`.
- By default, `llm_classify` loads this file; you may point `policy_path` to another bundle fetched by your own pipeline.

Runtime usage example
---------------------
```python
from agent_security_analyser.security_checker import llm_classify

decision = llm_classify(
    user_query="salary_view",
    user_ctx={"user_id": "alice", "role": "level_3", "location": "Munich"},
    resource_ctx={"project_location": "Munich", "is_owner_or_lead": False, "user_on_project": False, "target_resolved": True},
    model="gpt-4.1",
)
print(decision.status, decision.reason)
```

LLM classification example
--------------------------
```python
from agent_security_analyser.security_checker import llm_classify

decision = llm_classify(
    user_query="Show me salary of my team",
    user_ctx={"user_id": "alice", "role": "level_3", "location": "Munich"},
    resource_ctx={"project_location": "Munich", "is_owner_or_lead": False, "user_on_project": False, "target_resolved": True},
    model="gpt-4.1",
)
print(decision.status, decision.reason)
```

Debugging failing tasks
-----------------------
`agent_security_analyser/debug_failed_tasks.py` can classify known failure cases via `llm_classify`:
```bash
python -m agent_security_analyser.debug_failed_tasks \
  --classify-only \
  --tasks-path agent_security_analyser/plans/security_policy_failures.json \
  --model gpt-4.1
```
Outputs are saved under `agent_security_analyser/data/llm_security_decisions-<ts>.json`.

How to test the checker via debug_failed_tasks.py
-------------------------------------------------
1) Put `OPENAI_API_KEY` into `.env` (repo root) or export it in your shell.
2) Prepare/inspect a tasks file: default sample is `agent_security_analyser/plans/security_policy_failures.json` (each entry needs `task_id`, `spec_id`, `request`, `task_text`).
3) Run classification only:
```bash
python -m agent_security_analyser.debug_failed_tasks \
  --classify-only \
  --tasks-path agent_security_analyser/plans/security_policy_failures.json \
  --model gpt-4.1
```
4) Check console output for `[llm] <task_id> -> <status> (<reason>)` and open the saved report in `agent_security_analyser/data/llm_security_decisions-<ts>.json`.

If you need to regenerate the tasks JSON from an ERC3 session (read-only), omit `--classify-only` and pass `--benchmark/--workspace`; add `--classify-after` to classify right after fetching.

Notes
-----
- `policy_planner.py` and `dynamic_policy.py` are legacy/unused for policy delivery in the current flow.
- Rule-based classification has been removed; only the LLM path remains (via `llm_classify` or its `classify` alias).
- Ensure `.env` is present when relying on LLM classification; otherwise set `OPENAI_API_KEY` in the environment. 
