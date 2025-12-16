from typing import Any, Dict, List, Optional, Literal

from erc3 import erc3 as dev, ApiException, Erc3Client
from erc3.erc3 import ProjectDetail
from pydantic import BaseModel, Field

# Custom tool declarations grouped for reuse across agents.


class Req_DeleteWikiPage(BaseModel):
    tool: Literal["/wiki/delete"] = "/wiki/delete"
    file: str
    changed_by: Optional[dev.EmployeeID] = None


class Req_ListAllProjectsForUser(BaseModel):
    tool: Literal["/all-projects-for-user"] = "/all-projects-for-user"
    user: dev.EmployeeID


class Resp_ListAllProjectsForUser(BaseModel):
    lead_in: List[ProjectDetail]
    member_of: List[ProjectDetail]


class Req_ListAllCustomersForUser(BaseModel):
    tool: Literal["/all-customers-for-user"] = "/all-customers-for-user"
    user: dev.EmployeeID


class Resp_ListAllCustomersForUser(BaseModel):
    customers: List[dev.CompanyDetail]


class Req_RunSecurityCheck(BaseModel):
    """Call the manual security checker before executing potentially sensitive steps."""
    tool: Literal["/security/check"] = "/security/check"
    request: str = Field(..., description="Describe the action or user request to evaluate")
    # Pass optional JSON string (or omit) for user/resource context; dicts are also accepted server-side.
    user_ctx: Optional[str] = None
    resource_ctx: Optional[str] = None
    model: Optional[str] = None


class Resp_SecurityCheck(BaseModel):
    status: Literal["allow", "deny", "clarify"]
    reason: str
    user_ctx: Dict[str, Any]
    resource_ctx: Dict[str, Any]


# Wrap stock tools with clearer names to avoid confusing the LLM.
class GetTimesheetReportByProject(dev.Req_TimeSummaryByProject):
    pass


class GetTimesheetReportByEmployee(dev.Req_TimeSummaryByEmployee):
    pass


class CreateTimesheetEntryForUser(dev.Req_LogTimeEntry):
    pass


class Req_SearchProjectsEverywhere(dev.Req_SearchProjects):
    """Search projects across active and archived by default."""
    include_archived: bool = True


def list_my_projects(api: Erc3Client, user: str) -> Resp_ListAllProjectsForUser:
    page_limit = 32
    next_offset = 0
    lead_in = []
    member_of = []
    while True:
        try:
            prjs = api.search_projects(offset=next_offset, limit=page_limit, include_archived=True, team=dict(employee_id=user))

            for p in prjs.projects or []:
                detail = api.get_project(p.id).project
                role = [t for t in detail.team if t.employee == user][0].role

                if role == "Lead":
                    lead_in.append(detail)
                else:
                    member_of.append(detail)

            next_offset = prjs.next_offset
            if next_offset == -1:
                return Resp_ListAllProjectsForUser(lead_in=lead_in, member_of=member_of)
        except ApiException as e:
            if "page limit exceeded" in str(e):
                page_limit /= 2
                if page_limit <= 2:
                    raise


def list_my_customers(api: Erc3Client, user: str) -> Resp_ListAllCustomersForUser:
    page_limit = 32
    next_offset = 0
    loaded = []
    while True:
        try:
            custs = api.search_customers(offset=next_offset, limit=page_limit, account_managers=[user])

            for p in custs.companies or []:
                loaded.append(api.get_customer(p.id).company)

            next_offset = custs.next_offset
            if next_offset == -1:
                return Resp_ListAllCustomersForUser(customers=loaded)
        except ApiException as e:
            if "page limit exceeded" in str(e):
                page_limit /= 2
                if page_limit <= 2:
                    raise
