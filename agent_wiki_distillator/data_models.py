from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Literal
import argparse

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

class FilesCategories(BaseModel):
    file_name: str
    #file_content_category: list [Literal[ "other", "company_overview", "existed_employees", "existed_roles", "existed_systems", "existed_data", "existed_locations", "roles_and_sources_access_level" ] ]
    why: str


#     file_content_category: Literal["employee_access_rules" ("action_access", "data_read_access"), "external_bot_access_rules" ("action_access", "data_read_access"), "internal_bot_access_rules"("action_access", "data_read_access"), "roles_and_sources_access_level", "existed_roles", "existed_systems",  "existed_apis", "existed_data", "existed_locations"] 
# "access_rules",  "people_and_roles", "systems_and_overview", "action_access", "data_read_access", "locations", "apis"] 

#_____Entities_______
class SensitivityLevels(BaseModel):   # security_and_rules
    existed_types_of_sensitivity: list[str]

class RoleTypes(BaseModel):   # people_and_roles
    existed_role_names: list[str] = Field(..., description="CEO, engineer, team lead etc.")
    
class RoleLevels(BaseModel):   # people_and_roles
    existed_levels_of_role_hierarchy: list[str] = Field(..., description="Lvl1, lvl2. etc")
    
class SystemTypes(BaseModel):  # "systems_and_data"
    mentioned_systems_names: list[str] = Field(..., description="CRM, DataBase1, etc.")
 
class DataTypes(BaseModel):    # "systems_and_data" # apis
    mentioned_data_entities: list[str] = Field(..., description="emplyee skills, customer contacts, etc.")

class LocationTypes(BaseModel):  # locations
    existed_location_names: list[str] = Field(..., description="Toronto, New York, etc.")

class ActionTypes(BaseModel):   # security_and_rules # apis
    mentioned_possible_actions: list[str] = Field(..., description="read, update, write, search, etc.")

#_____Rules_1_______
class CompanyBlock(BaseModel):
    found_id: str = ""
    company_name: str = ""
    company_role: str = ""
    
class Rules(CompanyBlock):   # security_and_rules
    rules: list[EmployeeAccessRule | ExternalBotAccessRule | InternalBotAccessRule]


class Rule(BaseModel):   # security_and_rules
    type: Literal ["employee_access_rules", "external_bot_access_rules", "internal_bot_access_rules"]
    rule: str
    
class SensitivityRoleMandat(BaseModel): # security_and_rules
    role_level: Literal [RoleLevels.existed_levels_of_role_hierarchy.values]
    max_sensitivity_level_allowed: Literal [SensitivityLevels.existing_types_of_sensitivity.values]
    
class SensitivityDataMandat(BaseModel): # security_and_rules
    data_entity: Literal [DataTypes.mentioned_data_entities.values]
    sensitivity_level: Literal [SensitivityLevels.existing_types_of_sensitivity.values]
    
class SensitivitySystemMandat(BaseModel): # security_and_rules
    system: Literal [SystemTypes.mentioned_systems_names.values]
    sensitivity_level: Literal [SensitivityLevels.existing_types_of_sensitivity.values]
    
class EmployeeAccessRule(BaseModel): # security_and_rules
    type: Literal ["employee_access_rules"]
    role_level: Literal [RoleLevels.existed_levels_of_role_hierarchy.values]
    allowed: list[Transaction]
    deny: list[Transaction]
    
class Transaction(BaseModel): # security_and_rules
    actions: list[Literal[ActionTypes.mentioned_possible_actions.values]]
    data: list[Literal[DataTypes.mentioned_data_entities.values] ]



class ExternalBotAccessRule(BaseModel):
    type: Literal ["external_bot_access_rules"] 
    rule: str 

class InternalBotAccessRule(BaseModel):
    type: Literal ["internal_bot_access_rules"]
    rule: str 


    
#_____________Rules_v2______________
class CompanyBlock(BaseModel):
    found_id: str = ""
    company_name: str = ""
    company_role: str = ""

class AccessRules(BaseModel):
    actor: str = ""
    resorce_type: Literal ["system_acess", "data_acess"]
    resorce_name: str = ""
    action_allowed: str | None
    action_denied: str | None


class SecurityRule(BaseModel):
    path_and_row: str = Field(..., description="Path to file and line number")
    rule: str
    actors: List[str] = Field(default_factory=list, description="People/roles the rule applies to")
    data_action_scope: List[str] = Field(default_factory=list, description="Data or resources referenced")
    restrictions: List[str] = Field(default_factory=list, description="Allow/deny/conditions")


class SecurityExtraction(CompanyBlock):
    rules: List[SecurityRule] = Field(default_factory=list)


class LocationEntry(BaseModel):
    path: str
    location: str
    address: str | None = None
    contacts: List[str] = Field(default_factory=list)
    notes: str = ""


class LocationsExtraction(CompanyBlock):
    locations: List[LocationEntry] = Field(default_factory=list)


class PersonEntry(BaseModel):
    path: str
    name: str
    role: str
    location: str | None = None
    responsibilities: List[str] = Field(default_factory=list)
    reports_to: str | None = None


class PeopleExtraction(CompanyBlock):
    people: List[PersonEntry] = Field(default_factory=list)


class SystemEntry(BaseModel):
    path: str
    name: str
    description: str
    data_assets: List[str] = Field(default_factory=list)
    sensitivity: str | None = None
    integrations: List[str] = Field(default_factory=list)


class SystemsExtraction(CompanyBlock):
    systems: List[SystemEntry] = Field(default_factory=list)


class ApiEntry(BaseModel):
    path: str
    name: str
    purpose: str
    endpoints: List[str] = Field(default_factory=list)
    auth: str | None = None
    pii_fields: List[str] = Field(default_factory=list)


class ApisExtraction(CompanyBlock):
    apis: List[ApiEntry] = Field(default_factory=list)
    
    
    
    
#_____Rules_v0_generated_________________


class CompanyBlock(BaseModel):
    found_id: str = ""
    company_name: str = ""
    company_role: str = ""


class SecurityRule(BaseModel):
    path: str
    rule_summary: str
    actors: List[str] = Field(default_factory=list, description="People/roles the rule applies to")
    data_scope: List[str] = Field(default_factory=list, description="Data or resources referenced")
    restrictions: List[str] = Field(default_factory=list, description="Allow/deny/conditions")


class SecurityExtraction(CompanyBlock):
    rules: List[SecurityRule] = Field(default_factory=list)


class LocationEntry(BaseModel):
    path: str
    location: str
    address: str | None = None
    contacts: List[str] = Field(default_factory=list)
    notes: str = ""


class LocationsExtraction(CompanyBlock):
    locations: List[LocationEntry] = Field(default_factory=list)


class PersonEntry(BaseModel):
    path: str
    name: str
    role: str
    location: str | None = None
    responsibilities: List[str] = Field(default_factory=list)
    reports_to: str | None = None


class PeopleExtraction(CompanyBlock):
    people: List[PersonEntry] = Field(default_factory=list)


class SystemEntry(BaseModel):
    path: str
    name: str
    description: str
    data_assets: List[str] = Field(default_factory=list)
    sensitivity: str | None = None
    integrations: List[str] = Field(default_factory=list)


class SystemsExtraction(CompanyBlock):
    systems: List[SystemEntry] = Field(default_factory=list)


class ApiEntry(BaseModel):
    path: str
    name: str
    purpose: str
    endpoints: List[str] = Field(default_factory=list)
    auth: str | None = None
    pii_fields: List[str] = Field(default_factory=list)


class ApisExtraction(CompanyBlock):
    apis: List[ApiEntry] = Field(default_factory=list)