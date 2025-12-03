import os
from typing import List, Optional, Literal
import time

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from erc3 import store

from prompts import CRITERIA_SYSTEM_PROMPT
from data_models import SuccessCriteria

CLI_RED = "\x1B[31m"
CLI_GREEN = "\x1B[32m"
CLI_BLUE = "\x1B[34m"
CLI_YELLOW = "\x1B[33m"
CLI_CLR = "\x1B[0m"

# Load environment variables for API keys
load_dotenv()
from langfuse import get_client, observe

lf = get_client()
# --- Nebius config ---
NEBIUS_API_KEY = os.environ["NEBIUS_API_KEY"]
NEBIUS_API_BASE = "https://api.studio.nebius.com/v1/"

# --- OpenAI config (NON-Nebius) ---
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

MAX_COMPLETION_TOKENS=16384

# --- Clients ---
nebius_client = OpenAI(
    base_url=NEBIUS_API_BASE,
    api_key=NEBIUS_API_KEY,
)

openai_client = OpenAI(
    api_key=OPENAI_API_KEY,
)



def get_llm_client(provider: Literal["nebius", "openai"], model_id) -> tuple[OpenAI, str]:
    """Return the correct OpenAI-compatible client based on provider."""
    if provider == "openai":
        return openai_client, "openai/"+model_id # log in OpenRouter format
    if provider == "nebius":
        return nebius_client, model_id
    raise ValueError(f"Unknown provider: {provider}")


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

@observe(name="plan&criteria", as_type="generation")
def get_criteria(
    model_id: str,
    task_text: str,
    provider: Literal["nebius", "openai"] = "nebius",
) -> SuccessCriteria:
    """
    Phase 1: use chosen provider/model to extract success criteria.
    """
    try:
        print(f"{CLI_YELLOW}Phase 1: Defining Success State... (provider={provider}, model={model_id}){CLI_CLR}")
        client, model_for_log = get_llm_client(provider, model_id)

        completion = client.beta.chat.completions.parse(
            model=model_id,
            messages=[
                {
                    "role": "system",
                    "content": CRITERIA_SYSTEM_PROMPT,
                },
                {"role": "user", "content": task_text},
            ],
            response_format=SuccessCriteria,
        )
        return completion.choices[0].message.parsed
    except Exception as e:
        print(f"{CLI_RED}CRITICAL FAILURE in criteria generation: {e}{CLI_CLR}")
        raise e
    
    
@observe(as_type="generation", name="llm_step")
def run_llm_step(provider, model_id: str, current_log, response_format, task_id, api):
    client, model_for_log = get_llm_client(provider, model_id)

    started = time.time()
    completion = client.beta.chat.completions.parse(
        model=model_id,
        messages=current_log,
        response_format=response_format,
        max_completion_tokens=MAX_COMPLETION_TOKENS
    )
    api.log_llm(
            task_id=task_id,
            model=model_for_log, # must match slug from OpenRouter
            duration_sec=time.time() - started,
            usage=completion.usage,
        )
    
    return completion  # need to return completion for correct completion.usage logging in langfuse.