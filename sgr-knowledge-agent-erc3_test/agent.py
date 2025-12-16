import hashlib
import json
import os
import subprocess
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
from agent_wiki_distillator import policy_extractor, wiki_annotator
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

DOCS_ROOT = Path(__file__).resolve().parent / "docs"
FETCH_SCRIPT = Path(__file__).resolve().parent / "fetch_wiki.py"
DISTILL_CACHE_DIR = Path(__file__).resolve().parent / "extracted_data"
DISTILL_CACHE_META = DISTILL_CACHE_DIR / "distill_cache.json"
WIKI_INDEX_PATH = DOCS_ROOT / "wiki_index.json"


def _ensure_wiki_docs() -> None:
    if DOCS_ROOT.exists() and any(DOCS_ROOT.iterdir()):
        return
    DOCS_ROOT.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(FETCH_SCRIPT)], check=True)


def _ensure_wiki_index() -> dict:
    if WIKI_INDEX_PATH.exists():
        return json.loads(WIKI_INDEX_PATH.read_text(encoding="utf-8"))
    _ensure_wiki_docs()
    try:
        subprocess.run([sys.executable, str(FETCH_SCRIPT)], check=True)
    except Exception as exc:  # noqa: BLE001
        print(f"fetch_wiki failed: {exc}")
    if WIKI_INDEX_PATH.exists():
        return json.loads(WIKI_INDEX_PATH.read_text(encoding="utf-8"))
    return {}


def _wiki_sha(index: dict) -> str:
    sha = index.get("tree_sha1") or index.get("sha1") or ""
    if sha:
        return sha
    pieces = [f.get("sha1", "") for f in index.get("files", []) if isinstance(f, dict) and f.get("sha1")]
    if not pieces:
        return ""
    return hashlib.sha1("|".join(pieces).encode("utf-8")).hexdigest()


def _load_distill_cache() -> dict:
    if DISTILL_CACHE_META.exists():
        try:
            return json.loads(DISTILL_CACHE_META.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_distill_cache(meta: dict) -> None:
    DISTILL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    DISTILL_CACHE_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def ensure_policy_bundle(model: str, wiki_sha_hint: str | None = None) -> Path:
    """
    Ensure wiki is present locally, then distill policy bundle with wiki_annotator + policy_extractor.
    Re-runs distillation when wiki sha changes.
    """
    _ensure_wiki_docs()
    index = _ensure_wiki_index()
    wiki_sha = _wiki_sha(index)
    if wiki_sha_hint and wiki_sha_hint != wiki_sha:
        try:
            subprocess.run([sys.executable, str(FETCH_SCRIPT)], check=True)
            index = _ensure_wiki_index()
            wiki_sha = _wiki_sha(index)
        except Exception as exc:  # noqa: BLE001
            print(f"fetch_wiki on sha mismatch failed: {exc}")

    cache = _load_distill_cache()
    cached_path = Path(cache["policy_path"]) if cache.get("policy_path") else None
    if cache.get("wiki_sha") == wiki_sha and cached_path and cached_path.exists():
        return cached_path

    found, found_path = wiki_annotator.find_categories(
        model=model,
        index_path=WIKI_INDEX_PATH,
        docs_root=DOCS_ROOT,
        fetch_script=FETCH_SCRIPT,
    )
    bundle_path = policy_extractor.distill_policy_bundle(
        found_path=found_path,
        docs_root=DOCS_ROOT,
        output_dir=DISTILL_CACHE_DIR,
        wiki_sha=wiki_sha,
        model=model,
    )
    _save_distill_cache(
        {
            "wiki_sha": wiki_sha,
            "found_path": str(found_path),
            "policy_path": str(bundle_path),
        }
    )
    return bundle_path

# Tool do automatically distill wiki rules
def distill_rules(api: Erc3Client, llm: MyLLM, about: dev.Resp_WhoAmI) -> str:

    model_name = os.environ.get("OPENAI_MODEL", llm.model)

    policy_payload: dict = {}
    policy_path: Path | None = None
    try:
        policy_path = ensure_policy_bundle(model_name, about.wiki_sha1)
        policy_payload = json.loads(Path(policy_path).read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"policy distillation failed, trying fallback: {exc}")
        fallback_policy = Path(__file__).resolve().parents[1] / "agent_security_analyser" / "policy" / "manual_wiki_extracted_entities_copy.json"
        if fallback_policy.exists():
            policy_path = fallback_policy
            policy_payload = json.loads(fallback_policy.read_text(encoding="utf-8"))

    if policy_path and policy_path.exists():
        security_checker.DEFAULT_ENTITIES_PATH = policy_path
        try:
            security_checker._load_security_and_rules.__defaults__ = (policy_path,)
        except Exception:
            pass

    context_id = about.wiki_sha1
    loc = Path(f"context_{context_id}_v2.json")
    fallback_loc = Path(__file__).resolve().parent / "context_733815c19ae7c1d13f345a2b2a9aa13c67a74769_v2.json"

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

    distilled_cache: DistillWikiRules | None = None
    for cand in (loc, fallback_loc):
        if not cand.exists():
            continue
        try:
            raw_cache = json.loads(cand.read_text(encoding="utf-8"))
            if isinstance(raw_cache, dict) and "parsed" in raw_cache:
                distilled_cache = DistillWikiRules.model_validate(raw_cache["parsed"])
            elif isinstance(raw_cache, dict) and isinstance(raw_cache.get("content"), str):
                distilled_cache = DistillWikiRules.model_validate_json(raw_cache["content"])
            else:
                distilled_cache = DistillWikiRules.model_validate(raw_cache)
            break
        except Exception:
            continue

    company_section = policy_payload.get("company") or policy_payload
    company_name = (
        company_section.get("company_name")
        or policy_payload.get("company_name")
        or getattr(distilled_cache, "company_name", "the company")
    )
    raw_locations = (
        company_section.get("company_locations")
        or policy_payload.get("company_locations")
        or getattr(distilled_cache, "company_locations", [])
    )
    company_locations = []
    for loc in raw_locations or []:
        if isinstance(loc, str):
            company_locations.append(loc)
        elif isinstance(loc, dict):
            val = loc.get("company_location") or loc.get("location")
            if val:
                company_locations.append(val)
    company_execs = getattr(distilled_cache, "company_execs", [])

    role_levels = policy_payload.get("role_levels") or []
    security_structured = policy_payload.get("security_structured") or {}
    system_api_coverage = policy_payload.get("system_api_coverage") or {}

    raw_rules_section = policy_payload.get("rules") or policy_payload.get("security_and_rules") or {}
    raw_rules = raw_rules_section.get("rules") if isinstance(raw_rules_section, dict) else raw_rules_section
    relevant_categories: List[str] = ["other", "applies_to_guests" if about.is_public else "applies_to_users"]
    filtered_rules = []
    if isinstance(raw_rules, list):
        for r in raw_rules:
            if not isinstance(r, dict):
                continue
            cat = r.get("category")
            if cat and cat not in relevant_categories:
                continue
            filtered_rules.append(r)
    security_and_rules = {"rules": filtered_rules} if filtered_rules else {"rules": raw_rules or []}

    policy_block = {
        "company_name": company_name,
        "company_locations": company_locations,
        "role_levels": role_levels,
        "security_structured": security_structured,
        "security_and_rules": security_and_rules,
        "system_api_coverage": system_api_coverage,
    }
    manual_policy_json_api_system_new_rules = json.dumps(policy_block, indent=2)

    prompt = f"""You are AI Chatbot automating {company_name}.
    
Company locations: {company_locations}"""
    if company_execs:
        prompt += f"\nCompany execs: {company_execs}"

    prompt += """

Use available tools to execute task from the current user.

- To confirm project access - get or find project (and get after finding)
- When unsure about scope/sensitivity - run /security/check with the intended action and context
- Archival of entries or wiki deletion are not irreversible operations.
- Respond with proper Req_ProvideAgentResponse when:
    - Task is done
    - Task can't be completed (e.g. internal error, user is not allowed or clarification is needed)
- Make sure to always include ids of referenced entities in response links.
- if user might have access to a resource - double-chech that BEFORE denying
"""

    prompt += f"\n\n# Wiki distillation\n{manual_policy_json_api_system_new_rules}\n"
    prompt += "\n# Note\nAll security_and_rules are known to the security checker; access or deny can be clarified with them.\n"

    context_block = {
        "current_user_id": about.current_user,
        "location": about.location,
        "department": getattr(about, "department", None),
    }
    cleaned_context = {k: v for k, v in context_block.items() if v not in (None, "", [])}
    prompt += f"# Current context (trust it)\nToday date:{about.today}\nUser asked bot:{json.dumps(cleaned_context, ensure_ascii=False)}"

    if about.is_public:
        prompt += "\nCurrent actor is GUEST (Anonymous user)"
    else:
        employee = api.get_employee(about.current_user).employee
        employee.skills = []
        employee.wills = []
        dump = employee.model_dump_json()
        prompt += f"\n# Current actor is authenticated user. Got data about user_id by employee api: {employee.name}:\n{dump}"

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
        if last_query_entities and getattr(last_query_entities, "required_resources", None):
            resource_ctx.setdefault("required_resources", last_query_entities.required_resources)
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
        last_query_entities = None
        
        llm = MyLLM(api=api, model=model, task=task, max_tokens=32768, provider=provider)

        system_prompt = distill_rules(erc_client, llm, about)

        reason = Literal["security_violation", "request_not_supported_by_api", "possible_security_violation_check_project", "may_pass"]

        class RequestPreflightCheck(BaseModel):
            current_actor: str
            preflight_check_explanation_brief: Optional[str]
            denial_reason: reason
            outcome_confidence_1_to_5: Annotated[int, Gt(0), Lt(6)]

        # log will contain conversation context for the agent within task
        log = [{"role": "system", "content": system_prompt},
               {"role": "user", "content": f"Request: '{task.task_text}'"}
               ]

        try:
            expanded = llm.query_expansion(task.task_text)
            last_query_entities = expanded
            log.append({"role": "assistant", "content": f"Query entities: {expanded.model_dump_json(exclude_none=True)}"})
        except Exception as e:
            print(f"query_expansion failed: {e}")

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
