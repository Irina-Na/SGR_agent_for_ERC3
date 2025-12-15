from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field


DEFAULT_INDEX = Path("sgr-knowledge-agent-erc3_test/docs/wiki_index.json")
DEFAULT_DOCS_ROOT = Path("sgr-knowledge-agent-erc3_test/docs")
DEFAULT_FETCH_SCRIPT = Path("sgr-knowledge-agent-erc3_test/fetch_wiki.py")
FOUND_ROOT = Path("agent_wiki_extraction/found_data")

load_dotenv()


class FilesCategories(BaseModel):
    file_name: str
    file_content_category: list [Literal[ "other", "company_overview", "existed_employees", "existed_roles", "existed_systems", "existed_data", "existed_locations", "roles_and_sources_access_level" ] ]
    why: str


#     file_content_category: Literal["employee_access_rules" ("action_access", "data_read_access"), "external_bot_access_rules" ("action_access", "data_read_access"), "internal_bot_access_rules"("action_access", "data_read_access"), "roles_and_sources_access_level", "existed_roles", "existed_systems",  "existed_apis", "existed_data", "existed_locations"] 
# "access_rules",  "people_and_roles", "systems_and_overview", "action_access", "data_read_access", "locations", "apis"] 
