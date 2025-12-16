import json
import sys
from pathlib import Path
from typing import Annotated, List, Union, Literal, Optional
from annotated_types import MaxLen, MinLen, Gt, Lt
from pydantic import BaseModel, Field
from erc3 import erc3 as dev, ApiException, TaskInfo, ERC3, Erc3Client
from dotenv import load_dotenv
load_dotenv()


from langfuse import get_client

lf = get_client()

from tools import MyLLM
root_dir = str(Path(__file__).resolve().parents[1])
if root_dir not in sys.path:
    sys.path.append(root_dir)
from api_tools import (
    Req_DeleteWikiPage,
    Req_ListAllProjectsForUser,
    Req_ListAllCustomersForUser,
    Req_RunSecurityCheck,
    Resp_SecurityCheck,
    GetTimesheetReportByProject,
    CreateTimesheetEntryForUser,
    Req_SearchProjectsEverywhere,
    list_my_projects,
    list_my_customers,
)
from agent_security_analyser import security_checker

# next-step planner
class NextStep(BaseModel):
    current_state: str
    # we'll use only the first step, discarding all the rest.
    plan_remaining_steps_brief: Annotated[List[str], MinLen(1), MaxLen(5)] =  Field(..., description="explain your thoughts on how to accomplish - what steps to execute")
    # now let's continue the cascade and check with LLM if the task is done
    task_completed: bool
    # Routing to one of the tools to execute the first remaining step
    # if task is completed, model will pick ReportTaskCompletion
    first_step_from_plan: Union[
        dev.Req_ProvideAgentResponse,
        dev.Req_ListProjects,
        dev.Req_SearchProjects,
        Req_SearchProjectsEverywhere,
        Req_ListAllProjectsForUser,
        dev.Req_GetProject,
        dev.Req_UpdateProjectTeam,
        dev.Req_UpdateProjectStatus,
        dev.Req_ListEmployees,
        dev.Req_SearchEmployees,
        dev.Req_GetEmployee,
        dev.Req_UpdateEmployeeInfo,
        dev.Req_ListCustomers,
        Req_ListAllCustomersForUser,
        dev.Req_GetCustomer,
        dev.Req_SearchCustomers,
        dev.Req_SearchTimeEntries,
        GetTimesheetReportByProject,
        dev.Req_TimeSummaryByEmployee,
        dev.Req_GetTimeEntry,
        CreateTimesheetEntryForUser,
        dev.Req_UpdateTimeEntry,
        Req_DeleteWikiPage,
        Req_RunSecurityCheck,
    ] = Field(..., description="first step from plan above; use /security/check to validate sensitive steps")

CLI_RED = "\x1B[31m"
CLI_GREEN = "\x1B[32m"
CLI_BLUE = "\x1B[34m"
CLI_CLR = "\x1B[0m"

# Tool do automatically distill wiki rules
def distill_rules(api: Erc3Client, llm: MyLLM, about: dev.Resp_WhoAmI) -> str:

    context_id = about.wiki_sha1

    loc = Path(f"context_{context_id}_v2.json")
    fallback_loc = Path(__file__).resolve().parent / "context_733815c19ae7c1d13f345a2b2a9aa13c67a74769_v2.json"
    manual_policy_json_path = Path(__file__).resolve().parents[1] / "agent_security_analyser" / "policy" / "manual_wiki_extracted_entities_copy.json"
    manual_policy_json_api_system_new_rules = None
    if manual_policy_json_path.exists():
        manual_policy_data = json.loads(manual_policy_json_path.read_text(encoding="utf-8"))
        security_structured = manual_policy_data.get("security_structured", {})
        collected_sections = {}
        for key, source in [
            ("role_levels", manual_policy_data.get("role_levels")),
            ("sensitivity_role_mandats", security_structured.get("sensitivity_role_mandats")),
            ("sensitivity_data_mandats", security_structured.get("sensitivity_data_mandats")),
            ("system_api_coverage", manual_policy_data.get("system_api_coverage")),
            ("security_and_rules", manual_policy_data.get("security_and_rules")),
        ]:
            if source is not None:
                collected_sections[key] = source
        if collected_sections:
            manual_policy_json_api_system_new_rules = json.dumps(collected_sections, indent=2)

    Category = Literal["applies_to_guests", "applies_to_users", "other"]

    class Rule(BaseModel):
        why_relevant_summary: str = Field(...)
        category: Category = Field(...)
        compact_rule: str

    class DistillWikiRules(BaseModel):
        company_name: str
        company_locations: List[str] = Field(..., description="list of locations where company operates")
        company_execs: List[str]
        rules: List[Rule]

    if not loc.exists():
        if fallback_loc.exists():
            loc = fallback_loc
            '''
        print("New context discovered. Distilling rules once")
        schema = json.dumps(NextStep.model_json_schema())
        prompt = f"""
Carefully review the wiki below and identify most important security/scoping/data rules that will be highly relevant for the agent or user that are automating APIs of this company.

Pay attention to the rules that mention AI Agent or Public ChatBot. When talking about Public Chatbot use - applies_to_guests

Rules must be compact RFC-style, ok to use pseudo code for compactness. They will be used by an agent that operates following APIs: {schema}
""".strip()

        for path in api.list_wiki().paths:
            content = api.load_wiki(path)
            prompt += f"\n---- start of {path} ----\n\n{content}\n\n ---- end of {path} ----\n"


        messages = [{ "role": "system", "content": prompt}]

        # Persist only the parsed structure expected by DistillWikiRules.
        distilled = llm.query(messages, DistillWikiRules)
        distilled = distilled.choices[0].message.parsed

        loc.write_text(distilled.model_dump_json(indent=2), encoding="utf-8")
            '''
        else:
            raise FileNotFoundError(f"Expected distilled wiki cache at {loc}")
    raw_cache = json.loads(loc.read_text(encoding="utf-8"))
    if isinstance(raw_cache, dict) and "parsed" in raw_cache:
        distilled = DistillWikiRules.model_validate(raw_cache["parsed"])
    elif isinstance(raw_cache, dict) and isinstance(raw_cache.get("content"), str):
        try:
            distilled = DistillWikiRules.model_validate_json(raw_cache["content"])
        except Exception:
            distilled = DistillWikiRules.model_validate(raw_cache)
    else:
        distilled = DistillWikiRules.model_validate(raw_cache)

    prompt = f"""You are AI Chatbot automating {distilled.company_name}.
    
Company locations: {distilled.company_locations}
Company execs: {distilled.company_execs}

Use available tools to execute task from the current user.

- To confirm project access - get or find project (and get after finding)
- When unsure about scope/sensitivity - run /security/check with the intended action and context
- Archival of entries or wiki deletion are not irreversible operations.
- Respond with proper Req_ProvideAgentResponse when:
    - Task is done
    - Task can't be completed (e.g. internal error, user is not allowed or clarification is needed)
- Make sure to always include ids of referenced entities in response links.
- if user might have access to a resource - double-chech that BEFORE denying

# Rules
"""
    relevant_categories: List[Category] = ["other"]
    if about.is_public:
        relevant_categories.append("applies_to_guests")
    else:
        relevant_categories.append("applies_to_users")

    if manual_policy_json_api_system_new_rules:
        prompt += f"\n\n# Wiki distillation (manual policies)\n{manual_policy_json_api_system_new_rules}\n"
        prompt += "\n# Note\nAll security_and_rules are known to the security checker; access or deny can be clarified with them.\n"
    else:
        raise FileNotFoundError(f"Expected distilled wiki rules at {manual_policy_json_api_system_new_rules}")
        '''for r in distilled.rules:
            if r.category in relevant_categories:
                prompt += f"\n- {r.compact_rule}"
        '''
    # append at the end to keep rules in context cache
    prompt += f"# Current context (trust it)\nDate:{about.today}"

    if about.is_public:
        prompt += "\nCurrent actor is GUEST (Anonymous user)"
    else:
        employee = api.get_employee(about.current_user).employee
        employee.skills = []
        employee.wills = []
        dump = employee.model_dump_json()
        prompt += f"\n# Current actor is authenticated user: {employee.name}:\n{dump}"

    return prompt


def my_dispatch(client: Erc3Client, cmd: BaseModel, about: dev.Resp_WhoAmI):
    # example how to add custom tools or tool handling
    
    if isinstance(cmd, dev.Req_UpdateEmployeeInfo):
        # first pull
        cur = client.get_employee(cmd.employee).employee

        cmd.notes = cmd.notes or cur.notes
        cmd.salary = cmd.salary or cur.salary
        cmd.wills = cmd.wills or cur.wills
        cmd.skills = cmd.skills or cur.skills
        cmd.location = cmd.location or cur.location
        cmd.department = cmd.department or cur.department
        return client.dispatch(cmd)

    if isinstance(cmd, Req_DeleteWikiPage):
        return client.dispatch(dev.Req_UpdateWiki(content="", changed_by=cmd.changed_by, file=cmd.file))

    if isinstance(cmd, Req_ListAllProjectsForUser):
        return list_my_projects(client, cmd.user)

    if isinstance(cmd, Req_ListAllCustomersForUser):
        return list_my_customers(client, cmd.user)

    if isinstance(cmd, dev.Req_SearchProjects):
        cmd.include_archived = True
        return client.dispatch(cmd)

    if isinstance(cmd, dev.Req_ProvideAgentResponse):
        # drop link to current user
        cmd.links = [l for l in cmd.links if l.id != about.current_user]
        return client.dispatch(cmd)
    
    if isinstance(cmd, Req_RunSecurityCheck):
        # Build user context programmatically from who_am_i + employee record (ignore LLM-provided user_ctx).
        base_user_id = about.current_user or "guest"
        base_location = about.location
        role = "guest" if about.is_public else "level_3"
        employee = None
        if not about.is_public and about.current_user:
            try:
                employee = client.get_employee(about.current_user).employee
                base_location = base_location or getattr(employee, "location", None)
            except Exception:
                employee = None
        try:
            data = json.loads(security_checker.DEFAULT_ENTITIES_PATH.read_text(encoding="utf-8"))
            rules = data.get("security_structured", {}).get("employee_access_rules") or []
            emp_name = getattr(employee, "name", None)
            for r in rules:
                if r.get("employee_name") == emp_name:
                    lvl = (r.get("employee_level") or "").lower()
                    if "level 1" in lvl:
                        role = "level_1"
                    elif "level 2" in lvl:
                        role = "level_2"
                    elif "level 3" in lvl:
                        role = "level_3"
                    break
        except Exception:
            pass
        user_ctx = {"user_id": base_user_id, "role": role, "location": base_location}
        user_ctx = {k: v for k, v in user_ctx.items() if v is not None}

        # Allow resource_ctx as JSON string; default to dict.
        raw_resource_ctx = cmd.resource_ctx
        if isinstance(raw_resource_ctx, str):
            try:
                raw_resource_ctx = json.loads(raw_resource_ctx)
            except Exception:
                raw_resource_ctx = {}
        resource_ctx = raw_resource_ctx if isinstance(raw_resource_ctx, dict) else {}
        if about.location and "project_location" not in resource_ctx:
            resource_ctx["project_location"] = about.location
        resource_ctx.setdefault("target_resolved", False)
        try:
            decision = security_checker.llm_classify(cmd.request, user_ctx, resource_ctx, model=cmd.model)
        except Exception as e:
            return Resp_SecurityCheck(
                status="deny",
                reason=f"security_check_error:{e.__class__.__name__}",
                user_ctx=user_ctx,
                resource_ctx=resource_ctx,
            )
        return Resp_SecurityCheck(
            status=decision.status,
            reason=decision.reason,
            user_ctx=user_ctx,
            resource_ctx=resource_ctx,
        )
    return client.dispatch(cmd)

def run_agent(model: str, api: ERC3, task: TaskInfo, provider: Literal["nebius", "openai"]="nebius"):
    with lf.start_as_current_observation(as_type="span", name=task.spec_id) as root_span: 

        erc_client = api.get_erc_client(task)
        about = erc_client.who_am_i()
        
        llm = MyLLM(api=api, model=model, task=task, max_tokens=32768, provider=provider)

        system_prompt = distill_rules(erc_client, llm, about)

        reason = Literal["security_violation", "request_not_supported_by_api", "possible_security_violation_check_project", "may_pass"]

        class RequestPreflightCheck(BaseModel):
            current_actor: str
            preflight_check_explanation_brief: Optional[str]
            denial_reason: reason
            outcome_confidence_1_to_5: Annotated[int, Gt(0), Lt(6)]

        # log will contain conversation context for the agent within task
        log = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Request: '{task.task_text}'"},
        ]

        preflight_check = llm.query(log, RequestPreflightCheck).choices[0].message.parsed
        confidence = preflight_check.outcome_confidence_1_to_5

        if confidence >=4:
            print(f"PREFLIGHT {confidence}: {preflight_check.preflight_check_explanation_brief}")
            if preflight_check.denial_reason == "request_not_supported_by_api":
                erc_client.provide_agent_response("Not supported", outcome="none_unsupported")
                return
            if preflight_check.denial_reason == "security_violation":
                erc_client.provide_agent_response("Security check failed", outcome="denied_security")
                return

        log.append({"role": "system", "content": preflight_check.preflight_check_explanation_brief})

        # let's limit number of reasoning steps by 20, just to be safe
        for i in range(20):
            step = f"step_{i + 1}"
            print(f"Next {step}... ", end="")

            job = llm.query(log, NextStep).choices[0].message.parsed

            # print next sep for debugging
            print(job.plan_remaining_steps_brief[0], f"\n  {job.first_step_from_plan}")

            # Let's add tool request to conversation history as if OpenAI asked for it.
            # a shorter way would be to just append `job.model_dump_json()` entirely
            log.append({
                "role": "assistant",
                "content": f"state: {job.current_state}\nplan: {job.plan_remaining_steps_brief}\ntask_completed: {job.task_completed}",
                "tool_calls": [{
                    "type": "function",
                    "id": step,
                    "function": {
                        "name": job.first_step_from_plan.__class__.__name__,
                        "arguments": job.first_step_from_plan.model_dump_json(),
                    }}]
            })

            # now execute the tool by dispatching command to our handler
            try:
                result = my_dispatch(erc_client, job.first_step_from_plan, about)
                txt = result.model_dump_json(exclude_none=True, exclude_unset=True)
                print(f"{CLI_GREEN}OUT{CLI_CLR}: {txt}")
                txt = "DONE: " + txt
            except ApiException as e:
                txt = e.detail
                # print to console as ascii red
                print(f"{CLI_RED}ERR: {e.api_error.error}{CLI_CLR}")

                txt = "ERROR: " + txt

                # if SGR wants to finish, then quit loop
            if isinstance(job.first_step_from_plan, dev.Req_ProvideAgentResponse):
                print(f"{CLI_BLUE}agent {job.first_step_from_plan.outcome}{CLI_CLR}. Summary:\n{job.first_step_from_plan.message}")

                for link in job.first_step_from_plan.links:
                    print(f"  - link {link.kind}: {link.id}")
                break

            # and now we add results back to the convesation history, so that agent
            # we'll be able to act on the results in the next reasoning step.
            log.append({"role": "tool", "content": txt, "tool_call_id": step})
    lf.flush()
