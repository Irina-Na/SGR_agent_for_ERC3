import os
import time
from typing import List, Type, TypeVar, Literal

from dotenv import load_dotenv
from erc3 import ERC3, TaskInfo
from openai import OpenAI
from pydantic import BaseModel

# Load environment variables for API keys
load_dotenv()
from langfuse import get_client, observe

lf = get_client()

T = TypeVar('T', bound=BaseModel)

# --- Nebius/OpenAI config ---
NEBIUS_API_KEY = os.environ["NEBIUS_API_KEY"]
NEBIUS_API_BASE = "https://api.studio.nebius.com/v1/"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

nebius_client = OpenAI(base_url=NEBIUS_API_BASE, api_key=NEBIUS_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


def get_llm_client(provider: Literal["nebius", "openai"], model_id: str) -> tuple[OpenAI, str]:
    """Return OpenAI-compatible client and model name for logging."""
    if provider == "openai":
        if not openai_client:
            raise RuntimeError("OPENAI_API_KEY not set")
        return openai_client, "openai/" + model_id
    if provider == "nebius":
        return nebius_client, model_id
    raise ValueError(f"Unknown provider: {provider}")


class MyLLM:
    api: ERC3
    task: TaskInfo
    model: str
    max_tokens: int
    provider: Literal["nebius", "openai"]

    def __init__(self, api: ERC3, model: str, task: TaskInfo, max_tokens=40000, provider: Literal["nebius", "openai"]="nebius") -> None:
        self.api = api
        self.model = model
        self.task = task
        self.max_tokens = max_tokens
        self.provider = provider


    @observe(as_type="generation", name="llm_step")
    def query(self, messages: List, response_format: Type[T], model: str = None) -> T:
        client, model_for_log = get_llm_client(self.provider, model or self.model)

        started = time.time()
        resp = client.beta.chat.completions.parse(
            messages=messages,
            model=model or self.model,
            response_format=response_format,
            max_completion_tokens=self.max_tokens,
        )
        
        self.api.log_llm(
            task_id=self.task.task_id,
            model=model_for_log,
            duration_sec=time.time() - started,
            usage=resp.usage,
        )
        try:
            resp.choices[0].message.parsed
        except Exception as e:
            print(f"LLM parse error: {e}")
            print(f"Raw LLM message: {getattr(resp, 'content', resp)}")
            raise

        return resp
