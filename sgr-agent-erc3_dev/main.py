import textwrap
from openai import OpenAI
from erc3 import ERC3
from dotenv import load_dotenv
load_dotenv()

from agent import run_agent

client = OpenAI()
core = ERC3()
MODEL_ID = "openai/gpt-oss-120b" #"gpt-4.1"     # models for nebius here: https://tokenfactory.nebius.com/?modality=text2text
platform = "nebius" #  "openai" #   
version='0.1.0'

# Start session with metadata
res = core.start_session(
    benchmark="erc3-dev",
    workspace="my",
    name=f"NextStep SGR Agent ({MODEL_ID}) {version}",
    architecture="NextStep SGR Agent with OpenAI")

status = core.session_status(res.session_id)
print(f"Session has {len(status.tasks)} tasks")

for task in status.tasks:
    print("="*40)
    print(f"Starting Task: {task.task_id} ({task.spec_id}): {task.task_text}")
    # start the task
    core.start_task(task)
    try:
        run_agent(MODEL_ID, core, task, provider=platform)
    except Exception as e:
        print(e)
    result = core.complete_task(task)
    if result.eval:
        explain = textwrap.indent(result.eval.logs, "  ")
        print(f"\nSCORE: {result.eval.score}\n{explain}\n")

core.submit_session(res.session_id)











