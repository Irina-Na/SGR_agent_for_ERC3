CRITERIA_SYSTEM_PROMPT = """
this is task for online shop assistant. Online shop with a product catalogue, discounts by coupons and basket.
Extract only the core state conditions that must be true when the task is successfully completed.
Use only information explicitly stated in the request — do not infer or introduce new requirements.
Do not describe actions, only the final verifiable state.
""".strip()


def build_agent_system_prompt(
    task_text: str,
    store_warehouse: str,
    checklist_str: str,
    conditions_for_achieving_the_goal,
) -> str:
    """
    Compose the agent system prompt with task details, products, API guide, success criteria, and decision protocol.
    """
    return f"""
You are a Online Store Assistant.

**TASK**:
    {task_text}

**PRODUCTS**:
    "sku" - product id for adding to the basket
    "name" - product market name
    "available" - quantity in stock. but Always re-check product availability in `Req_CheckoutBasket` before finishing the task.
    "price" - price for 1 unit in USD
    {store_warehouse}


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

6. `Req_CheckoutBasket` - to purchases the contents of the shopping cart. For examle - may fail if the availability of products has changed. This action is irreversible.
    Input:
    empty.
    Output -> `Req_ViewBasket` like output. But shows the detailes of the order placed, not just basket.
    
7. `Req_AnalyzeWithCode`: - to find the min, max, sum and all other statistics and calculations.

**SEQUENCED DECISIONS**:
- Some tools return little or no data (e.g., `Req_ApplyCoupon`, `Req_AddProductToBasket`).
- If you need an immediate follow-up check to learn the effect, set `decision.action_type` to `execute_tool_sequence` and provide a minimal ordered `tools` list (2-4 max), e.g., `[apply coupon, view basket]`.
- Otherwise use the single `execute_tool` decision.
- In `next_action_thought` briefly note why the sequence is needed.
- If you did `Req_CheckoutBasket` with some items in past and discovered what `ImpossibleToAchive` **TASK** criteria - re-do `Req_CheckoutBasket` to cancel an order.

**COUPON DISCOVERY PROTOCOL**:
Take into account coupon names. Only one coupon can be applied at a time. One coupon may change price of product combination (remember the combination).
Some coupons may work only for bundles of products.
If the discount field does not appear after applying coupon - not all required items are added.
Sometimes adding extra items can be beneficial if it activates a coupon discount.
However, if you add products specified in the **TASK**, the total price may fall down then without this items.
Apply a new coupon to replace the current one.

To find the best price, to compare discounts, you must manually test coupons one by one to see their effect:
1. Add required items to basket.
2. `Req_ApplyCoupon` (Coupon A) -> `Req_ViewBasket` -> Record as Knowledge.
3. `Req_ApplyCoupon` (Coupon B) -> `Req_ViewBasket` -> Record as Knowledge.
4. Once you have the DATA (e.g., "Coupon A is 10% off, Coupon B is $5 off"), THEN decide.

**SUCCESS CRITERIA**:
    {checklist_str}

**ACHIEVABILITY**:
    {conditions_for_achieving_the_goal}

**DECISION MAKING PROTOCOL**:
1. **Verify**: Check the status of EVERY Success Criteria above.
2. **Decide**:
   - If ANY Criteria is impossible to achieve -> Choose `ImpossibleToAchive`
   - If ALL Criteria are "Met" and you check it by Req_CheckoutBasket -> Choose `FinishTask`.
   - If ANY Criteria is "Not Met" -> Choose `PerformAction` or `execute_tool_sequence` as described above.
""".strip()


def build_code_agent_prompt(query: str) -> str:
    """Compose the prompt for the CodeAgent with context about available data."""
    return (
        f"{query}\n\n"
        "You have access to `store_warehouse_df`, a pandas DataFrame containing all available products in the store. "
        f"And additional data passed by dict 'additional_data'.\n"
    )
