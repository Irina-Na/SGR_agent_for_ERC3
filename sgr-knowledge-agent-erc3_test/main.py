import textwrap

from agent import run_agent
from erc3 import ERC3


core = ERC3()
# Use the actual OpenAI model id; the API rejects the prefixed variant.
MODEL_ID = "gpt-4.1"
PLATFORM = "openai"  # or "nebius"
version='0.1.0'


# Debugging a single task
# task = core.start_new_task("erc3-test", "project_check_by_member")
#run_agent(MODEL_ID, core, task)



# Start session with metadata
res = core.start_session(
    benchmark="erc3-dev", #test",
    workspace="ira",
    name=f"NextStep SGR ({MODEL_ID}) {version} from ERC3 Samples +pipelined",
    architecture="NextStep SGR Agent with OpenAI")

status = core.session_status(res.session_id)
print(f"Session has {len(status.tasks)} tasks")

for task in status.tasks:
    print("="*40)
    print(f"Starting Task: {task.task_id} ({task.spec_id}): {task.task_text}")
    # start the task
    core.start_task(task)
    try:
        run_agent(MODEL_ID, core, task, provider=PLATFORM)
    except Exception as e:
        print(e)
    result = core.complete_task(task)
    if result.eval:
        explain = textwrap.indent(result.eval.logs, "  ")
        print(f"\nSCORE: {result.eval.score}\n{explain}\n")

core.submit_session(res.session_id)












