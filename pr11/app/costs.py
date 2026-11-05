import os
from dotenv import load_dotenv

load_dotenv()

PRICE_INPUT = float(os.getenv("LLM_PRICE_INPUT_PER_1M", "0.30"))
PRICE_OUTPUT = float(os.getenv("LLM_PRICE_OUTPUT_PER_1M", "2.50"))
_cached = os.getenv("LLM_PRICE_CACHED_INPUT_PER_1M", "")
PRICE_CACHED_INPUT = float(_cached) if _cached else PRICE_INPUT * 0.25
MOCK_BILLABLE = os.getenv("MOCK_BILLABLE", "1") == "1"


def settings() -> dict:
    return {
        "input_per_1m": PRICE_INPUT,
        "output_per_1m": PRICE_OUTPUT,
        "cached_input_per_1m": PRICE_CACHED_INPUT,
        "mock_billable": MOCK_BILLABLE,
        "currency": "USD",
    }


def estimate(usage: dict | None, *, provider: str | None = None, model: str | None = None,
             text: str | None = None) -> tuple[float | None, bool]:
    if provider == "mock" and not MOCK_BILLABLE:
        return 0.0, False

    if usage and isinstance(usage, dict):
        cached_tokens = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
        prompt_tokens = usage.get("prompt_tokens", 0) or 0
        regular_input = max(0, prompt_tokens - cached_tokens)
        completion_tokens = usage.get("completion_tokens", 0) or 0
        cost = (regular_input / 1e6 * PRICE_INPUT) + (cached_tokens / 1e6 * PRICE_CACHED_INPUT) + (completion_tokens / 1e6 * PRICE_OUTPUT)
        return round(cost, 6), False

    if text:
        est_in = 500
        est_out = max(1, len(text) // 3)
        cost = (est_in / 1e6 * PRICE_INPUT) + (est_out / 1e6 * PRICE_OUTPUT)
        return round(cost, 6), True

    return 0.0, True
