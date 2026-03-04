import textwrap
import time

from agent import run_agent
from erc3 import ERC3, ApiException
import os

core = ERC3()
# Use the actual OpenAI model id; the API rejects the prefixed variant.
MODEL_ID = "gpt-4.1"
PLATFORM = "openai"  # or "nebius"
version='0.3.1'
BENCHMARK = "erc3-prod" # "erc3-test"  # set once to keep fetch_wiki in sync

os.environ.setdefault("ERC3_BENCHMARK", BENCHMARK)
os.environ.setdefault("ERC3_WORKSPACE", "ira")

# Debugging a single task
# task = core.start_new_task("erc3-test", "project_check_by_member")
#run_agent(MODEL_ID, core, task)



# Start session with metadata
res = core.start_session(
    benchmark=BENCHMARK, #test",
    workspace=os.environ["ERC3_WORKSPACE"],
    name=f"NextStep SGR ({MODEL_ID}) {version} + json_entities_wiki_distillation+pipelined",
    architecture="NextStep SGR Agent from ERC3 Samples with OpenAI + json_entities_wiki_distillation+api_system_match+fix_unmatched_apis+new_rules extractor+security_checker as tool+adds wrapped_apis ",
    flags=["sgr", "compete_speed", "compete_budget"],
     )

status = core.session_status(res.session_id)
print(f"Session has {len(status.tasks)} tasks")

def _complete_with_retry(api: ERC3, task, attempts: int = 3, delay: float = 1.5):
    for i in range(1, attempts + 1):
        try:
            return api.complete_task(task)
        except ApiException as e:
            print(f"complete_task failed (attempt {i}/{attempts}): {e}")
            if i == attempts:
                raise
            time.sleep(delay)
        except Exception as e:
            print(f"complete_task unexpected error (attempt {i}/{attempts}): {e}")
            if i == attempts:
                raise
            time.sleep(delay)

for task in status.tasks:
    print("="*40)
    print(f"Starting Task: {task.task_id} ({task.spec_id}): {task.task_text}")
    # start the task
    core.start_task(task)
    try:
        run_agent(MODEL_ID, core, task, provider=PLATFORM)
    except Exception as e:
        print(e)
    result = _complete_with_retry(core, task)
    if result.eval:
        explain = textwrap.indent(result.eval.logs, "  ")
        print(f"\nSCORE: {result.eval.score}\n{explain}\n")

core.submit_session(res.session_id)










