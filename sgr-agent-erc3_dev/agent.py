from typing import Literal
from erc3 import erc3 as dev, ApiException, TaskInfo, ERC3

from tools import get_criteria,  run_llm_step
from prompts import build_agent_system_prompt
from data_models import NextStep

CLI_RED = "\x1B[31m"
CLI_GREEN = "\x1B[32m"
CLI_BLUE = "\x1B[34m"
CLI_CLR = "\x1B[0m"


from langfuse import get_client

lf = get_client()

def run_agent(model: str, api: ERC3, task: TaskInfo, provider: Literal["nebius", "openai"] = "nebius"):
    """
    Run the store agent loop.

    provider = "nebius"  — uses Nebius endpoint with Nebius models
    provider = "openai"  — uses OpenAI endpoint with OpenAI models (e.g. gpt-4.1, gpt-4o, o3-mini)
    """
    store_api = api.get_erc_dev_client(task)
    about = store_api.who_am_i()

    # нужна инициализация головного трейса на уровне run_agent, чтобы все возможные вызовы по задаче собирать в один трейс
    with lf.start_as_current_observation(as_type="span", name=task.spec_id) as root_span: 
        root_span.update_trace(name=task.spec_id)
        
        system_prompt = build_agent_system_prompt(user_info=about.model_dump_json())
        if about.current_user:
            usr = store_api.get_employee(about.current_user)
            system_prompt += f"\n{usr.model_dump_json()}"

        # log will contain conversation context for the agent within task
        log = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task.task_text},
        ]


            
        # let's limit number of reasoning steps by 20, just to be safe
        for i in range(20):
            step = f"step_{i + 1}"
            print(f"Next {step}... ", end="")

            completion = run_llm_step(provider, model, log, NextStep, task.task_id, api)
            
            job = completion.choices[0].message.parsed  # need to return completion for correct completion.usage logging in langfuse.

            # print next sep for debugging
            print(job.plan_remaining_steps_brief[0], f"\n  {job.function}")
            
            # Let's add tool request to conversation history as if OpenAI asked for it.
            # a shorter way would be to just append `job.model_dump_json()` entirely
            log.append({
                "role": "assistant",
                "content": job.plan_remaining_steps_brief[0],
                "tool_calls": [{
                    "type": "function",
                    "id": step,
                    "function": {
                        "name": job.function.__class__.__name__,
                        "arguments": job.function.model_dump_json(),
                    }}]
            })

            # now execute the tool by dispatching command to our handler
            try:
                result = store_api.dispatch(job.function)
                txt = result.model_dump_json(exclude_none=True, exclude_unset=True)
                print(f"{CLI_GREEN}OUT{CLI_CLR}: {txt}")
            except ApiException as e:
                txt = e.detail
                # print to console as ascii red
                print(f"{CLI_RED}ERR: {e.api_error.error}{CLI_CLR}")

                # if SGR wants to finish, then quit loop
            if isinstance(job.function, dev.Req_ProvideAgentResponse):
                print(f"{CLI_BLUE}agent {job.function.outcome}{CLI_CLR}. Summary:\n{job.function.message}")

                for link in job.function.links:
                    print(f"  - link {link.kind}: {link.id}")

                break

            # and now we add results back to the convesation history, so that agent
            # we'll be able to act on the results in the next reasoning step.
            log.append({"role": "tool", "content": txt, "tool_call_id": step})

    lf.flush()