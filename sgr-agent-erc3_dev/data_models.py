from pydantic import BaseModel, Field
from typing import Annotated, List, Union, Literal
from annotated_types import MaxLen, MinLen
from erc3 import erc3 as dev




#_______ FINAL DATA MODEL FOR SRC-3 DEV AGENT OUTPUT ON EVERY STEP _______#

class NextStep(BaseModel):
    current_state: str
    # we'll use only the first step, discarding all the rest.
    plan_remaining_steps_brief: Annotated[List[str], MinLen(1), MaxLen(5)] =  Field(..., description="explain your thoughts on how to accomplish - what steps to execute")
    # now let's continue the cascade and check with LLM if the task is done
    task_completed: bool
    # Routing to one of the tools to execute the first remaining step
    # if task is completed, model will pick ReportTaskCompletion
    function: Union[
        dev.Req_ProvideAgentResponse,
        dev.Req_ListProjects,
        dev.Req_ListEmployees,
        dev.Req_ListCustomers,
        dev.Req_GetCustomer,
        dev.Req_GetEmployee,
        dev.Req_GetProject,
        dev.Req_GetTimeEntry,
        dev.Req_SearchProjects,
        dev.Req_SearchEmployees,
        dev.Req_LogTimeEntry,
        dev.Req_SearchTimeEntries,
        dev.Req_SearchCustomers,
        dev.Req_UpdateTimeEntry,
        dev.Req_UpdateProjectTeam,
        dev.Req_UpdateProjectStatus,
        dev.Req_UpdateEmployeeInfo,
        dev.Req_TimeSummaryByProject,
        dev.Req_TimeSummaryByEmployee,
    ] = Field(..., description="execute first remaining step")




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
    trick: str  = Field(..., description="Suggest what else can be tried to find more successful ways to achieve the goal.") 
