import textwrap

from store_agent import run_agent, NEBIUS_API_KEY, NEBIUS_API_BASE
from erc3 import ERC3

# Initialize ERC3 Core
core = ERC3()

platform = "openai"  # "nebius" #"openai"
MODEL_ID = "gpt-4.1"
CRITERIA_MODEL_ID = "gpt-5.1"

def main():
    # Start session with metadata
    res = core.start_session(
        benchmark="store",
        workspace="ira",
        name=f"{platform} {MODEL_ID} {CRITERIA_MODEL_ID} + parser + ref knowledge",
        architecture="SGR Agent + code agent + Added data about API + store parser",
    )

    print(f"Session ID: {res.session_id}")
    
    # Get tasks
    status = core.session_status(res.session_id)
    print(f"Session has {len(status.tasks)} tasks")

    for task in status.tasks:
        print("="*60)
        print(f"Starting Task: {task.task_id} ({task.spec_id})")
        print(f"Instruction: {task.task_text}")
        print("-" * 60)
        
        # start the task
        core.start_task(task)
        
        try:
  
            run_agent(MODEL_ID, CRITERIA_MODEL_ID, core, task , provider=platform)
        except Exception as e:
            print(f"CRITICAL FAILURE: {e}")
            # Optional: Fail the task explicitly if needed, 
            # though usually we let the agent report failure.
        
        # Complete and Score
        result = core.complete_task(task)
        if result.eval:
            explain = textwrap.indent(result.eval.logs, "  ")
            print(f"\nSCORE: {result.eval.score}\n{explain}\n")

    # Submit results
    core.submit_session(res.session_id)
    print("Session submitted.")

if __name__ == "__main__":
    main()
