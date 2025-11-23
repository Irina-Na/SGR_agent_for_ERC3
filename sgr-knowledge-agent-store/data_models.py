from typing import List, Union, Literal, Optional

from pydantic import BaseModel, Field
from erc3 import store


class SuccessCriteria(BaseModel):
    success_criteria: List[str] = Field(
        ...,
        description="List of state assertions (e.g., 'Item X is in basket').",
    )
    conditions_for_achieving_the_goal: List[str] = Field(
        ...,
        description="What conditions must be met for achieving the goal?",
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


class KnowledgeItem(BaseModel):
    fact_or_move_and_result: str = Field(
        ...,
        description=(
            "An important fact learned during the task execution and API call. "
            "E.g., '4�-6pk: $4 off (SAVE10); 24pk: $18 off (FIT20).'"
        ),
    )


class NextMove(BaseModel):
    knowledges: List[KnowledgeItem] = Field(
        ...,
        description=("Save all useful facts gathered throughout the last step."),
    )
    state_assessment: List[CriterionState]
    thought_process: Optional[str] = Field(None, description=("Very short"))
    decision: Union[PerformAction, ImpossibleToAchive, FinishTask]


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
