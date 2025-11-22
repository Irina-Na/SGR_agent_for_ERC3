import time
import json
from typing import Annotated, List, Union, Literal, Optional
from annotated_types import MaxLen, MinLen
from pydantic import BaseModel, Field
import time
import os
from erc3 import store, ApiException, TaskInfo, ERC3

# AI Imports
from openai import OpenAI
from smolagents import CodeAgent, OpenAIServerModel

# ==========================================
# 1. CONFIGURATION
# ==========================================


from dotenv import load_dotenv
load_dotenv()

# --- Nebius config ---
NEBIUS_API_KEY = os.environ["NEBIUS_API_KEY"]

NEBIUS_API_BASE = "https://api.studio.nebius.com/v1/"

# --- OpenAI config (NON-Nebius) ---

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# SLIDING WINDOW CONFIG
CONTEXT_WINDOW_SIZE = 3  # Keep only last N tool interactions

# --- Clients ---
nebius_client = OpenAI(
    base_url=NEBIUS_API_BASE,
    api_key=NEBIUS_API_KEY,
)

openai_client = OpenAI(
    api_key=OPENAI_API_KEY,  # Default base URL → OpenAI API
)

def get_llm_client(provider: Literal["nebius", "openai"]) -> OpenAI:
    """Return the correct OpenAI-compatible client based on provider."""
    if provider == "openai":
        return openai_client
    if provider == "nebius":
        return nebius_client
    raise ValueError(f"Unknown provider: {provider}")


# ==========================================
# 2. DATA MODELS
# ==========================================

class InitialPlan(BaseModel):
    core_goal: str
    success_criteria: List[str] = Field(
        ...,
        description="List of state assertions (e.g., 'Item X is in basket')."
    )

class CriterionState(BaseModel):
    requirement: str
    status: Literal["Met", "Not Met"]
    evidence: str = Field(..., description="Quote tool output proving this.")

class Req_AnalyzeWithCode(BaseModel):
    query: str

class PerformAction(BaseModel):
    """Select this to execute a store tool or code analysis."""
    action_type: Literal["execute_tool"]
    tool: Union[
        store.Req_ListProducts,
        store.Req_ViewBasket,
        store.Req_ApplyCoupon,
        store.Req_RemoveCoupon,
        store.Req_AddProductToBasket,
        store.Req_RemoveItemFromBasket,
        store.Req_CheckoutBasket,
        Req_AnalyzeWithCode
    ] = Field(..., description="The specific tool to run.")

class FinishTask(BaseModel):
    """Select this ONLY if ALL success criteria are 'Met'."""
    action_type: Literal["finish_task"]
    final_summary: str
    code: Literal["completed", "failed"]

class NextMove(BaseModel):
    knowledge_summary: str = Field(
        ...,
        description=(
            "A summary of IMPORTANT facts gathered so far. "
            "E.g., '6pk is $4, 24pk is $18. Coupon SAVE10 is active.'"
        )
    )
    state_assessment: List[CriterionState]
    thought_process: str
    decision: Union[PerformAction, FinishTask] = Field(
        ...,
        description="Do you need to perform an action or are you finished?"
    )


# ==========================================
# 3. LOGIC
# ==========================================

CLI_RED = "\x1B[31m"
CLI_GREEN = "\x1B[32m"
CLI_BLUE = "\x1B[34m"
CLI_YELLOW = "\x1B[33m"
CLI_CLR = "\x1B[0m"


def get_criteria(
    model_id: str,
    task_text: str,
    provider: Literal["nebius", "openai"] = "nebius",
) -> InitialPlan:
    """Phase 1: use chosen provider/model to extract success criteria."""
    print(f"{CLI_YELLOW}Phase 1: Defining Success State... (provider={provider}, model={model_id}){CLI_CLR}")
    client = get_llm_client(provider)

    completion = client.beta.chat.completions.parse(
        model=model_id,
        messages=[
            {
                "role": "system",
                "content": '''
Extract only the core state conditions that must be true when the task is successfully completed.
List no more than 3 conditions.
Use only information explicitly stated in the request — do not infer or introduce new requirements.
Do not describe actions, only the final verifiable state.
                '''.strip(),
            },
            {"role": "user", "content": task_text},
        ],
        response_format=InitialPlan,
    )
    return completion.choices[0].message.parsed


def run_agent(
    model_id: str,
    api: ERC3,
    task: TaskInfo,
    provider: Literal["nebius", "openai"] = "nebius",
):
    """
    Run the store agent loop.

    provider = "nebius"  → uses Nebius endpoint with Nebius models
    provider = "openai"  → uses OpenAI endpoint with OpenAI models (e.g. gpt-4.1, gpt-4o, o3-mini)
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
            api_key=OPENAI_API_KEY,  # default api_base → OpenAI
            max_tokens=10000,
        )

    code_agent = CodeAgent(
        tools=[],
        model=smol_model,
        additional_authorized_imports=["math", "datetime", "re"],
    )

    # 1. Plan
    plan = get_criteria(model_id, task.task_text, provider=provider)
    checklist_str = "\n".join([f"- {c}" for c in plan.success_criteria])
    print(f"Goal: {plan.core_goal}")
    print(f"Assertions:\n{checklist_str}\n")

    system_prompt = f"""
You are a Store Agent.

**Goal**: {plan.core_goal}
**Assertions**:
{checklist_str}

**TOOL USAGE GUIDE (Read Carefully)**:
1. `Req_ListProducts`: Use to find items, check prices, and see inventory.
2. `Req_AddProductToBasket`: Use ONLY for adding physical products. **NEVER use this for coupons.**
3. `Req_RemoveItemFromBasket`: Use to remove product from basket.
4. `Req_ApplyCoupon`: Use ONLY for applying discount codes (e.g. "SAVE10", "FIT20").
5. `Req_RemoveCoupon`: Use to remove coupon.
6. `Req_ViewBasket`: Use to check if items are added and if coupons are active.
7. `Req_CheckoutBasket`: Use only at the very end to finalize the purchase. You can purchase only ONCE! So make it count.
8. `Req_AnalyzeWithCode`: Use for optimisations. **IMPORTANT**: You CANNOT ask code to compare coupons if you do not know what they do.

**COUPON DISCOVERY PROTOCOL**:
To find the best price, you must manually test coupons one by one to see their effect:
1. Add items to basket.
2. `Req_ApplyCoupon` (Coupon A) -> `Req_ViewBasket` -> Record Total.
3. `Req_ApplyCoupon` (Coupon B) -> `Req_ViewBasket` -> Record Total.
4. Once you have the DATA (e.g., "Coupon A is 10% off, Coupon B is $5 off"), THEN decide.

**Protocol**:
1. **Verify**: Check the status of EVERY assertion above.
2. **Decide**:
   - If ANY assertion is "Not Met" -> Choose `PerformAction`.
   - If ANY assertion is impossible to achieve based on new data -> Choose `FinishTask`
   - If ALL assertions are "Met" -> Choose `FinishTask`.

**Store hints**:
- If ListProducts returns non-zero "NextOffset", it means there are more products available.
- You can apply coupon codes using `Req_ApplyCoupon` to get discounts.
- Some coupouns may work with bundles. Take into account coupon names.
- Best way to gather info about possible product combos for coupons - add product types in question to basket, apply coupons and look how prices change. One coupon may change price of product combination.
- Use ViewBasket to see current discount and total.
- Only one coupon can be applied at a time. Apply a new coupon to replace the current one, or remove it explicitly.
- If you want to compare discounts, first you will have to collect information about product prices with applied coupons.
""".strip()

    # Base log (system + initial request)
    base_log = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"ORIGINAL REQUEST:\n{task.task_text}\n\n"
                "Begin execution. Verify the state of the store first."
            ),
        },
    ]

    # Sliding window for recent interactions
    recent_interactions: List[
        tuple[dict, dict]
    ] = []  # List of (assistant_msg, tool_output) tuples

    # Knowledge accumulator - carries forward between iterations
    accumulated_knowledge = ""

    for i in range(30):

        step_label = f"Step {i+1}"
        print(f"{step_label}: Thinking...", end=" ")
        started = time.time()

        # Build current context: base + knowledge summary + recent window
        current_log = base_log.copy()

        # Inject accumulated knowledge before recent interactions
        if accumulated_knowledge:
            current_log.append(
                {
                    "role": "user",
                    "content": (
                        "ACCUMULATED KNOWLEDGE FROM PREVIOUS STEPS:\n"
                        f"{accumulated_knowledge}"
                    ),
                }
            )

        for assistant_msg, tool_output in recent_interactions:
            current_log.append(assistant_msg)
            current_log.append(tool_output)

        completion = client.beta.chat.completions.parse(
            model=model_id,
            messages=current_log,
            response_format=NextMove,
        )
                # ---- FAILURE DETECTION & RETRY ----
        raw_content = completion.choices[0].message.content or ""
        if "CRITICAL FAILURE" in raw_content.upper():
            print(f"{CLI_RED}!! MODEL FAILURE DETECTED → waiting 10 seconds and retrying...{CLI_CLR}")
            time.sleep(10)
            continue

        # Log with provider prefix
        api.log_llm(
            task_id=task.task_id,
            model=f"{provider}/{model_id}",
            duration_sec=time.time() - started,
            usage=completion.usage,
        )

        move = completion.choices[0].message.parsed

        # Update accumulated knowledge from this turn
        accumulated_knowledge = move.knowledge_summary

        # Log Decision Type
        met_count = sum(1 for c in move.state_assessment if c.status == "Met")
        decision_type = "Action" if isinstance(move.decision, PerformAction) else "Finish"

        print(f"\n[Knowledge]: {move.knowledge_summary}")
        print(f"[State]: {met_count}/{len(move.state_assessment)} Met -> {decision_type}")
        print(f"[Thought]: {move.thought_process}")

        # --- COMPLETION HANDLER ---
        if isinstance(move.decision, FinishTask):
            unmet = [c.requirement for c in move.state_assessment if c.status == "Not Met"]

            # Guardrail: Anti-Hallucination
            if move.decision.code == "completed" and unmet:
                print(f"{CLI_RED}GUARDRAIL: Rejected. Unmet: {unmet[0]}...{CLI_CLR}")

                # Generate a specific hint
                hint = "You must continue working."
                if "inventory" in unmet[0].lower() or "identified" in unmet[0].lower():
                    hint = "Call `Req_ListProducts` to verify inventory."
                elif "added" in unmet[0].lower() or "basket" in unmet[0].lower():
                    hint = "Call `Req_AddProductToBasket`."

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

            print(f"{CLI_BLUE}Finished: {move.decision.code}{CLI_CLR}")
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
                res = store_api.dispatch(tool_obj)
                tool_output = res.model_dump_json(exclude_none=True, exclude_unset=True)
                print(f"  {CLI_GREEN}<< API OK{CLI_CLR}")

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

        recent_interactions.append((assistant_msg, tool_output_msg))

        # Keep only last N interactions
        if len(recent_interactions) > CONTEXT_WINDOW_SIZE:
            recent_interactions.pop(0)  # Remove oldest interaction
