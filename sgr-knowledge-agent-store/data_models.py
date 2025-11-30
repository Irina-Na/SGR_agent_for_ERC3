from typing import List, Union, Literal, Optional

from pydantic import BaseModel, Field, ConfigDict
from erc3 import store




#_______ STORE TOOLS DATA MODELS _______#
class Req_AnalyzeWithCode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., description="Short calculation or statistics request.")
    additional_data: Optional[str] = Field(
        None,
        description="Optional serialized data (stringified JSON) that may be useful for calculations, except store warehouse list.",
    )



class BasketItem(BaseModel):
    sku: str = Field(..., description="Product id to add into the basket.")
    quantity: int = Field(..., gt=0, description="How many units to add.")


class CheckCoupon(BaseModel):
    """Input payload for `check_coupon`."""

    coupon: str = Field(..., description="Coupon code to apply.")
    items: List[BasketItem] = Field(
        ...,
        description="Items that should be in the basket before applying coupon.",
    )



#_______ DATA MODELS FOR STORE AGENT DECISIONS _______#
class ImpossibleToAchive(BaseModel):
    """Select this if any criteria is impossible to achieve."""

    action_type: Literal["impossible_to_achieve"]
    reason: str = Field(..., description="Short explanation")

class FinishTask(BaseModel):
    """Select this ONLY if ALL success criteria are 'Met'."""

    action_type: Literal["finish_task"]


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


class PerformActionSequence(BaseModel):
    """Select this to execute a minimal ordered sequence of tools in one step."""

    action_type: Literal["execute_tool_sequence"]
    tools: List[
        Union[
            store.Req_ViewBasket,
            store.Req_ApplyCoupon,
            store.Req_RemoveCoupon,
            store.Req_AddProductToBasket,
            store.Req_RemoveItemFromBasket,
            Req_AnalyzeWithCode,
            store.Req_CheckoutBasket,
        ]
    ] = Field(
        ...,
        description=(
            "Run these tools in order as one decision. "
            "Use only when the earlier tool gives little or no immediate information "
            "and must be paired with a follow-up check (e.g., apply coupon -> view basket)."
        ),
    )


#_______ DATA MODELS FOR STORE AGENT Success TRACKING _______#
class SuccessCriteria(BaseModel):
    success_criteria: List[str] = Field(
        ...,
        description="List of state assertions (e.g., 'Item X is in basket').",
    )
    conditions_for_achieving_the_goal: List[str] = Field(
        ...,
        description="What conditions must be met for achieving the goal?",
    )


class CriterionState(BaseModel):
    criteria_id: str
    thought_about_achievement: str = Field(..., description="Review all your knowledges. Confirm you’ve checked the additional conditions listed in the criterion to achieve maximum effect.")
    status: Literal["Met", "Not Met"]
    trick: str | None = Field(None, description="If criteria not meet - suggest what else can be tried to find more successful ways to achieve the goal.") 

    
#_______ DATA MODELS FOR STORE AGENT KNOWLEDGE TRACKING _______#

class KnowledgeItem(BaseModel):
    fact_or_move_and_result: str = Field(
        ...,
        description=(
            "An important fact learned during the task execution and API call. "
            "E.g., '4×6pk: $4 off (SAVE10); 24pk: $18 off (FIT20).'"
        ),
    )


#_______ FINAL DATA MODEL FOR STORE AGENT OUTPUT ON EVERY STEP _______#
class NextMove(BaseModel):
    knowledge: List[KnowledgeItem] = Field(
        ...,
        description=("Save all useful facts gathered throughout the last step."),
    )
    state_assessment: List[CriterionState]
    next_action_thought: str = Field(...,
        description=(
            "Brief rationale (<2 sentences). Note when a minimal tool sequence is needed "
            "because an earlier unary action gives no useful info (e.g., apply coupon -> view basket)."
        ),
    )
    decision: Union[ImpossibleToAchive, PerformAction, PerformActionSequence, FinishTask]

