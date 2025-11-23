import time
import json
from typing import Annotated, List, Union, Literal, Optional
from pydantic import BaseModel, Field
import time
import os
import pandas as pd
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


def fetch_available_products_list(
    store_api,
    page_size: Optional[int] = None,
) -> pd.DataFrame:
    """
    Fetch all products via Req_ListProducts using a single store client and
    return only items that are in stock as a pandas DataFrame.

    - If page_size is None (default), uses limit=0 to ask the API for the
      maximum allowed page in one call.
    - If page_size is set, paginates using next_offset and the provided limit
      to avoid exceeding API limits.
    """
    products: List[dict] = []

    limit = 0 if page_size is None else page_size
    offset = 0

    while True:
        response = store_api.dispatch(
            store.Req_ListProducts(limit=limit, offset=offset)
        )

        for p in response.products:
            if p.available and p.available > 0:
                products.append(
                    {
                        "sku": p.sku,
                        "name": p.name,
                        "price": p.price,
                        "available": p.available,
                    }
                )
        if response.next_offset <= 0:
            break
        limit = len(response.products)
        offset += limit

    return products

# ==========================================
# 2. DATA MODELS
# ==========================================

class SuccessCriteria(BaseModel):
    success_criteria: List[str] = Field(
        ...,
        description="List of state assertions (e.g., 'Item X is in basket')."
    )
    conditions_for_achieving_the_goal: List[str] = Field(
        ...,
        description="What conditions must be met for achieving the goal?"
    )
    
class ImpossibleToAchive(BaseModel):
    """Select this if any criteria is impossible to achieve."""
    action_type: Literal["impossible_to_achieve"]
    reason: str = Field(..., description="Short explanation")
    
class CriterionState(BaseModel):
    criteria_id: str
    status: Literal["Met", "Not Met"]
    evidence: str = Field(..., description="Quote tool output proving this.")

class Req_AnalyzeWithCode(BaseModel):
    query: str

class PerformAction(BaseModel):
    """Select this to execute a store tool or code analysis."""
    action_type: Literal["execute_tool"]
    tool: Union[
        store.Req_ViewBasket,
        store.Req_ApplyCoupon,
        store.Req_RemoveCoupon,
        store.Req_AddProductToBasket,
        store.Req_RemoveItemFromBasket,
        Req_AnalyzeWithCode,
        store.Req_CheckoutBasket,
    ] = Field(..., description="The specific tool to run.")

class FinishTask(BaseModel):
    """Select this ONLY if ALL success criteria are 'Met'."""
    action_type: Literal["finish_task"]


class NextMove(BaseModel):
    knowledges: List[str] = Field(
        ...,
        description=(
            "Save all IMPORTANT facts gathered throughout the entire process of solving the problem."
            "E.g., '6pk is $4, 24pk is $18. Coupon SAVE10 is active.'"
        )
    )
    state_assessment: List[CriterionState]
    thought_process: Optional[str] = Field(None, description=("Very short" ))
    decision: Union[PerformAction, FinishTask, ImpossibleToAchive]


# ==========================================
# 3. LOGIC
# ==========================================

CLI_RED = "\x1B[31m"
CLI_GREEN = "\x1B[32m"
CLI_BLUE = "\x1B[34m"
CLI_YELLOW = "\x1B[33m"
CLI_CLR = "\x1B[0m"

def get_api_call(store_api, tool_obj):
    try:
        res = store_api.dispatch(tool_obj)
        tool_output = res.model_dump_json(exclude_none=True, exclude_unset=True)
        print(f"\n [Tool Output]: {tool_output}")
        print(f"  {CLI_GREEN}<< API OK{CLI_CLR}")
        return tool_output
    except Exception as e:
        err = f"Checkout failed: {e}"
        print(f"{CLI_RED}{err}{CLI_CLR}")
        return err


def get_criteria(
    model_id: str,
    task_text: str,
    provider: Literal["nebius", "openai"] = "nebius",
) -> SuccessCriteria:
    try:
        """Phase 1: use chosen provider/model to extract success criteria."""
        print(f"{CLI_YELLOW}Phase 1: Defining Success State... (provider={provider}, model={model_id}){CLI_CLR}")
        client = get_llm_client(provider)

        completion = client.beta.chat.completions.parse(
            model=model_id,
            messages=[
                {
                    "role": "system",
                    "content": '''this is task for online shop assistant. Online shop with a product catalogue, discounts and basket. 
    Extract only the core state conditions that must be true when the task is successfully completed.   
    Use only information explicitly stated in the request — do not infer or introduce new requirements.
    Do not describe actions, only the final verifiable state.
                    '''.strip(),
                },
                {"role": "user", "content": task_text},
            ],
            response_format=SuccessCriteria,
        )
        return completion.choices[0].message.parsed
    except Exception as e:
        print(f"{CLI_RED}CRITICAL FAILURE in criteria generation: {e}{CLI_CLR}")
        raise e


def run_agent(
    model_id: str,
    get_criteria_model_id: str,
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

    store_warehouse = fetch_available_products_list(store_api)
    print(f"[Store Warehouse]:\n{('\n'.join([str(p) for p in store_warehouse]))}\n")
    
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
    plan = get_criteria(get_criteria_model_id, task.task_text, provider=provider)
    checklist_str = "\n".join([f"id:{i} {c}" for i, c in enumerate(plan.success_criteria)])

    print(f"Criteria:\n{checklist_str}\n")
    print("Conditions for achieving the goal:")
    for cond in plan.conditions_for_achieving_the_goal:
        print(f" - {cond}")

    system_prompt = f"""
You are a Online Store Assistant.

**TASK**: 
    {task.task_text}

**PRODUCTS**: 
    "sku" - product id for adding to the basket
    "name" - product market name
    "available" - quantity in stock. but Always re-check product availability in `Req_CheckoutBasket` before finishing the task.
    "price" - price for 1 unit in USD
    {str(store_warehouse)}


**API - TOOL USAGE GUIDE (Read Carefully)**:

1. `Req_AddProductToBasket`:  for adding physical products to basket.
    Input:
    "sku" - product id
    "quantity" - how much to add to the basket
    Output:
    To check the final availability, use `Req_CheckoutBasket`.

2. `Req_RemoveItemFromBasket`: to remove product from basket.
    Input:
    "sku" - product id
    "quantity" - how much to remove from the basket
    Output:
    To check the final availability, use `Req_CheckoutBasket`.

3. `Req_ApplyCoupon`: to apply discount codes (e.g. "SAVE10", "FIT20") for all sku in basket.
    Input:  
    "coupon" - name of coupon
    Output:
    empty response. To check the effect, use `Req_ViewBasket`.

4. `Req_RemoveCoupon`: to remove one coupon for all sku in basket. Better - just apply a new coupon to replace the current one.
    Input:  
    "coupon" - name of coupon
    Output:
    empty response. To check the effect, use `Req_ViewBasket`.

5. `Req_ViewBasket`: to check what coupons are applied and their effects.
    Input:  
    empty.
    Output:
    "items": [
        "price" - price per unit,
        "quantity" - how many units,
        "sku" - product id,
            ],
    "subtotal" - for all items before discount,
    "coupon" - Optional, name of active coupon
    "total" - after discount,
    "discount" - Optional - total discount in USD. Exist only if coupon realy gives discount.

6. `Req_CheckoutBasket` - to finalize the purchase and re-check product availability in real-time.
    Input:
    empty.

7. `Req_AnalyzeWithCode`: use for any calculations. **IMPORTANT**: You CANNOT ask code to compare coupons if you do not know what they do.

**COUPON DISCOVERY PROTOCOL**:
Take into account coupon names. Only one coupon can be applied at a time. One coupon may change price of product combination.
Some coupons may work only for bundles of products.
Best way to gather info about possible product combos for coupons - add product types from **TASK** to basket, apply coupons and look how prices change.
If the combination does not fit conditions of the coupon, coupon will not work (the price after applying will be the same). However, if you add certain products specified in the **TASK**, the price may change.
Apply a new coupon to replace the current one.
If you want to compare discounts, first you will have to collect information about product prices with applied coupons.

To find the best price, you must manually test coupons one by one to see their effect:
1. Add items to basket.
2. `Req_ApplyCoupon` (Coupon A) -> `Req_ViewBasket` -> Record as Knowledge.
3. `Req_ApplyCoupon` (Coupon B) -> `Req_ViewBasket` -> Record as Knowledge.
4. Once you have the DATA (e.g., "Coupon A is 10% off, Coupon B is $5 off"), THEN decide.

**SUCCESS CRITERIA**: 
    {checklist_str}
    
**ACHIEVABILITY**:
    {plan.conditions_for_achieving_the_goal}
    
**DECISION MAKING PROTOCOL**:
1. **Verify**: Check the status of EVERY Success Criteria above.
2. **Decide**:
   - If ANY Criteria is impossible to achieve -> Choose `ImpossibleToAchive`
   - If ANY Criteria is "Not Met" -> Choose `PerformAction`.
   - If ALL Criteria are "Met" and you check it by Req_CheckoutBasket -> Choose `FinishTask`.
""".strip()

    # Knowledge accumulator - carries forward between iterations
    accumulated_knowledge = ""
    
    # Base log (system + initial request)
    base_log = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"ORIGINAL REQUEST:\n{task.task_text}\n\n"
                "Begin execution. Verify the state of the store first."
                f"KNOWLEDGE ACCUMULATION: {"\n".join(accumulated_knowledge)}"
            ),
        },
    ]

    # Sliding window for recent interactions
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
            model=f"{provider}/{model_id}", # todo: add criteria model?
            duration_sec=time.time() - started,
            usage=completion.usage,
        )

        move = completion.choices[0].message.parsed

        # Update accumulated knowledge from this turn
        accumulated_knowledge = move.knowledges

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
