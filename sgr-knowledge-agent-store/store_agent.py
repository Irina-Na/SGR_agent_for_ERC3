import time
import json
from typing import List, Literal

from erc3 import store, TaskInfo, ERC3
from data_models import (
    SuccessCriteria,
    ImpossibleToAchive,
    CriterionState,
    Req_AnalyzeWithCode,
    PerformAction,
    FinishTask,
    KnowledgeItem,
    NextMove,
    BasketItem,
    CheckCoupon,
)
from prompts import build_agent_system_prompt
from tools import (
    fetch_available_products_list,
    get_api_call,
    check_coupon,
    get_criteria,
    get_llm_client,
    CLI_RED,
    CLI_GREEN,
    CLI_BLUE,
    CLI_YELLOW,
    CLI_CLR,
    NEBIUS_API_BASE,
    NEBIUS_API_KEY,
    OPENAI_API_KEY,
)

# AI Imports
from smolagents import CodeAgent, OpenAIServerModel

from langfuse import get_client, observe
langfuse = get_client()


# SLIDING WINDOW CONFIG
CONTEXT_WINDOW_SIZE = 3  # Keep only last N tool interactions


@observe()
def run_agent(
    model_id: str,
    get_criteria_model_id: str,
    api: ERC3,
    task: TaskInfo,
    provider: Literal["nebius", "openai"] = "nebius",
    run_name: str | None = None,
):
    """
    Run the store agent loop.

    provider = "nebius"  — uses Nebius endpoint with Nebius models
    provider = "openai"  — uses OpenAI endpoint with OpenAI models (e.g. gpt-4.1, gpt-4o, o3-mini)
    """
    client = get_llm_client(provider)
    store_api = api.get_store_client(task)

    # ---- Model for smolagents CodeAgent ----
    if provider == "nebius":
        smol_model = OpenAIServerModel(
            model_id=model_id,
            api_base=NEBIUS_API_BASE,
            api_key=NEBIUS_API_KEY,
            max_tokens=10000,
        )
    else:  # "openai"
        smol_model = OpenAIServerModel(
            model_id=model_id,
            api_key=OPENAI_API_KEY,  # default api_base — OpenAI
            max_tokens=10000,
        )

    code_agent = CodeAgent(
        tools=[],
        model=smol_model,
        additional_authorized_imports=["math", "datetime", "re"],
    )

    # 1. Plan
    plan = get_criteria(
        get_criteria_model_id,
        task.task_text,
        provider=provider,
    )
    checklist_str = "\n".join([f"id:{i} {c}" for i, c in enumerate(plan.success_criteria)])

    print(f"Criteria:\n{checklist_str}\n")
    print("Conditions for achieving the goal:")
    for cond in plan.conditions_for_achieving_the_goal:
        print(f" - {cond}")

    # 2. Store Warehouse (API Schema + Data)
    store_warehouse, current_view_basket = fetch_available_products_list(store_api)
    print(f"[Store Warehouse]:\n{('\n'.join([str(p) for p in store_warehouse]))}\n")
    
    # 3. Build System Prompt
    system_prompt = build_agent_system_prompt(
        task.task_text,
        str(store_warehouse),
        checklist_str,
        plan.conditions_for_achieving_the_goal,
    )

    # 4. Knowledge accumulator - carries forward between iterations
    accumulated_knowledge = [f'basket_state: {current_view_basket}']
        
    # 5. Base log (system + initial request)
    base_log = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"ORIGINAL REQUEST:\n{task.task_text}\n\n"
                f"PREVIOUS STEPS: {'\\n'.join([str(item) for item in accumulated_knowledge])}\n\n"
                f"CRITERIA:\n{checklist_str}"
            ),
        },
    ]

    # 6. Sliding window for recent interactions
    recent_interactions: List[
        tuple[dict, dict]
    ] = []  # List of (assistant_msg, tool_output) tuples

    # Build current context: base + knowledge summary + recent window

    log = []
    for i in range(20):

        step_label = f"Step {i+1}"
        print(f"{step_label}: Thinking...", end=" ")
        started = time.time()
            
        for assistant_msg, tool_output in recent_interactions:
            log.append(assistant_msg)
            log.append(tool_output)
        
        current_log = base_log.copy()
        current_log = current_log + log
        
        
        langfuse.update_current_trace(name=task.spec_id)
        # Log the current conversation context for observability
        langfuse.update_current_span(input={"messages": current_log})


        completion = client.beta.chat.completions.parse(
            model=model_id,
            messages=current_log,
            response_format=NextMove,
        )
        
        langfuse.update_current_span(output=completion)
        # ---- FAILURE DETECTION & RETRY ----
        raw_content = completion.choices[0].message.content or ""
        if "CRITICAL FAILURE" in raw_content.upper():
            print(f"{CLI_RED}!! MODEL FAILURE DETECTED — waiting 10 seconds and retrying...{CLI_CLR}")
            time.sleep(10)
            continue

        # Log with provider prefix
        api.log_llm(
            task_id=task.task_id,
            model=f"{provider}/{model_id}", # todo: add criteria model?
            duration_sec=time.time() - started,
            usage=completion.usage,
        )

        move = completion.choices[0].message.parsed

        # Update accumulated knowledge from this turn
        accumulated_knowledge.append(move.knowledges)

        # Log Decision Type
        met_count = sum(1 for c in move.state_assessment if c.status == "Met")
        decision_type = "Action" if isinstance(move.decision, PerformAction) else "Finish"

        print(f"\n[Knowledge]: {accumulated_knowledge}")
        print(f"[State]: {met_count}/{len(move.state_assessment)} Met -> {decision_type}")
        print(f" {move.state_assessment}")
        print(f"[Thought]: {move.thought_process}")

        # --- COMPLETION HANDLER ---
        if isinstance(move.decision, ImpossibleToAchive):
            print(f"{CLI_RED}Task impossible to achieve: {move.decision.reason}{CLI_CLR}")
            
            break
            
        if isinstance(move.decision, FinishTask):
            unmet = [c.requirement for c in move.state_assessment if c.status == "Not Met"]

            # Guardrail: Anti-Hallucination
            if unmet:
                print(f"{CLI_RED}GUARDRAIL: Rejected. Unmet: {unmet[0]}...{CLI_CLR}")

                # Generate a specific hint
                hint = "You must continue working."
                
                # Add to sliding window
                assistant_msg = {
                    "role": "assistant",
                    "content": json.dumps(move.model_dump(mode="json")),
                }
                tool_output_msg = {
                    "role": "user",
                    "content": (
                        f"SYSTEM ERROR: Criteria not met ({unmet[0]}). {hint}"
                    ),
                }

                recent_interactions.append((assistant_msg, tool_output_msg))
                if len(recent_interactions) > CONTEXT_WINDOW_SIZE:
                    recent_interactions.pop(0)  # Remove oldest

                continue

            # Complete the task
            print(f"{CLI_BLUE}Finished: {move.decision}{CLI_CLR}")
            break

        # --- ACTION HANDLER ---
        action = move.decision
        tool_obj = action.tool
        tool_name = tool_obj.__class__.__name__

        print(f"  >> Executing {tool_name}")

        # Execute
        tool_output = ""
        try:
            if isinstance(tool_obj, Req_AnalyzeWithCode):
                print(f"  {CLI_BLUE}>> CodeAgent: {tool_obj.query}{CLI_CLR}")
                res = code_agent.run(tool_obj.query)
                tool_output = f"Analysis: {res}"
                print(f"  {CLI_GREEN}<< Code OK{CLI_CLR}")
            else:
                tool_output = get_api_call(store_api, tool_obj)
                print(f"[Tool Output]: {tool_output}")
                
        except Exception as e:
            tool_output = f"Error: {str(e)}"
            print(f"  {CLI_RED}<< {tool_output}{CLI_CLR}")

        # Add to sliding window
        assistant_msg = {
            "role": "assistant",
            "content": json.dumps(move.model_dump(mode="json")),
        }
        tool_output_msg = {
            "role": "user",
            "content": f"Tool Output: {tool_output}",
        }

        recent_interactions = [(assistant_msg, tool_output_msg)]

        # Keep only last N interactions
        if len(recent_interactions) > CONTEXT_WINDOW_SIZE:
            recent_interactions.pop(0)  # Remove oldest interaction
