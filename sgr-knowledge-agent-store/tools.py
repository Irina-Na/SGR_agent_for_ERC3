import os
from pathlib import Path
from typing import List, Optional, Literal

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from erc3 import store
from langfuse import observe
from data_models import CheckCoupon, SuccessCriteria
from prompts import CRITERIA_SYSTEM_PROMPT

CLI_RED = "\x1B[31m"
CLI_GREEN = "\x1B[32m"
CLI_BLUE = "\x1B[34m"
CLI_YELLOW = "\x1B[33m"
CLI_CLR = "\x1B[0m"

# Load environment variables for API keys.
# Ensure we always load the package-local .env (contains Langfuse keys) even if the
# process is started from the repo root that has a different .env without Langfuse.
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR.parent / ".env")
load_dotenv(BASE_DIR / ".env", override=False)

# --- Nebius config ---
NEBIUS_API_KEY = os.environ["NEBIUS_API_KEY"]
NEBIUS_API_BASE = "https://api.studio.nebius.com/v1/"

# --- OpenAI config (NON-Nebius) ---
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# --- Clients ---
nebius_client = OpenAI(
    base_url=NEBIUS_API_BASE,
    api_key=NEBIUS_API_KEY,
)

openai_client = OpenAI(
    api_key=OPENAI_API_KEY,
)


def get_llm_client(provider: Literal["nebius", "openai"]) -> OpenAI:
    """Return the correct OpenAI-compatible client based on provider."""
    if provider == "openai":
        return openai_client
    if provider == "nebius":
        return nebius_client
    raise ValueError(f"Unknown provider: {provider}")


def fetch_available_products_list(
    store_api,
    page_size: Optional[int] = None,
) -> tuple[List[dict], dict]:    
    """
    Fetch all products via Req_ListProducts using a single store client and
    return only items that are in stock as a pandas DataFrame.

    - If page_size is None (default), uses limit=0 to ask the API for the
      maximum allowed page in one call.
    - If page_size is set, paginates using next_offset and the provided limit
      to avoid exceeding API limits.
    """
    products: List[dict] = []

    limit = 0 if page_size is None else page_size
    offset = 0

    while True:
        response = store_api.dispatch(
            store.Req_ListProducts(limit=limit, offset=offset)
        )

        for p in response.products:
            if p.available and p.available > 0:
                products.append(
                    {
                        "sku": p.sku,
                        "name": p.name,
                        "price": p.price,
                        "available": p.available,
                    }
                )
        if response.next_offset <= 0:
            break
        limit = len(response.products)
        offset += limit

    basket_state = store_api.dispatch(store.Req_ViewBasket())

    return products, basket_state


def get_api_call(store_api, tool_obj):
    try:
        res = store_api.dispatch(tool_obj)
        tool_output = res.model_dump_json(exclude_none=True, exclude_unset=True)
        print(f"{CLI_GREEN}<< API OK{CLI_CLR}")
        print(f"[Tool Output]: {tool_output}\n")

        return tool_output
    except Exception as e:
        err = f"Checkout failed: {e}"
        print(f"{CLI_RED}{err}{CLI_CLR}")
        # Bubble up so the agent loop stops the sequence and surfaces the failure.
        raise


def check_coupon(store_api, payload: CheckCoupon):
    """
    Clear the basket, add requested items, apply the coupon, and return basket state.
    """
    basket_state = store_api.dispatch(store.Req_ViewBasket())

    for item in getattr(basket_state, "items", []) or []:
        if item.quantity <= 0:
            continue
        store_api.dispatch(
            store.Req_RemoveItemFromBasket(sku=item.sku, quantity=item.quantity)
        )

    if getattr(basket_state, "coupon", None):
        try:
            store_api.dispatch(store.Req_RemoveCoupon(coupon=basket_state.coupon))
        except Exception:
            pass

    for item in payload.items:
        store_api.dispatch(
            store.Req_AddProductToBasket(sku=item.sku, quantity=item.quantity)
        )

    if payload.coupon:
        store_api.dispatch(store.Req_ApplyCoupon(coupon=payload.coupon))

    final_basket = store_api.dispatch(store.Req_ViewBasket())
    response = {
        "items": [
            {"sku": i.sku, "quantity": i.quantity, "price": i.price}
            for i in getattr(final_basket, "items", []) or []
        ],
        "subtotal": getattr(final_basket, "subtotal", 0),
        "total": getattr(final_basket, "total", 0),
    }

    if getattr(final_basket, "coupon", None):
        response["coupon"] = final_basket.coupon
    discount_value = getattr(final_basket, "discount", None)
    if discount_value:
        response["discount"] = discount_value

    return response

@observe()
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
        client = get_llm_client(provider)

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
