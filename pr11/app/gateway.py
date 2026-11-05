import logging
import os
import random
import threading
import time
from typing import Callable
from dotenv import load_dotenv

from . import budget, cache, costs, telemetry
from .providers.base import Chunk, ProviderError, Request, Result
from .providers.mock import MockProvider
from .providers.openai_compat import OpenAICompatProvider

load_dotenv()

PROVIDER = os.getenv("LLM_PROVIDER", "openai")
FALLBACK = os.getenv("LLM_FALLBACK", "none")
TIMEOUT = float(os.getenv("LLM_TIMEOUT", "20"))
IDLE_TIMEOUT = float(os.getenv("LLM_IDLE_TIMEOUT", "8"))
TOTAL_TIMEOUT = float(os.getenv("LLM_TOTAL_TIMEOUT", "45"))
MAX_ATTEMPTS = int(os.getenv("LLM_MAX_ATTEMPTS", "3"))
BACKOFF_BASE = float(os.getenv("LLM_BACKOFF_BASE", "0.5"))
BACKOFF_MAX = float(os.getenv("LLM_BACKOFF_MAX", "8"))
MAX_CONCURRENT = int(os.getenv("LLM_MAX_CONCURRENT", "2"))
MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")

log = logging.getLogger("app.gateway")
_semaphore = threading.Semaphore(MAX_CONCURRENT)


def settings() -> dict:
    return {
        "provider": os.getenv("LLM_PROVIDER", PROVIDER),
        "fallback": os.getenv("LLM_FALLBACK", FALLBACK),
        "timeout": TIMEOUT,
        "idle_timeout": IDLE_TIMEOUT,
        "total_timeout": TOTAL_TIMEOUT,
        "max_attempts": MAX_ATTEMPTS,
        "max_concurrent": MAX_CONCURRENT,
        "model": MODEL,
    }


def _create_provider(name: str):
    if name == "mock":
        return MockProvider()
    base_url = os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
    api_key = os.getenv("LLM_API_KEY", "")
    model = os.getenv("LLM_MODEL", MODEL)
    return OpenAICompatProvider(base_url=base_url, api_key=api_key, model=model, timeout=TIMEOUT)


def generate_stream(
    request: Request,
    on_chunk: Callable[[str], None],
    is_cancelled: Callable[[], bool],
) -> Result:
    prompt_version = request.metadata.get("prompt_version", "")
    draft_id = request.metadata.get("draft_id", "")
    client_id = request.metadata.get("client", "anonymous")
    req_id = request.metadata.get("http_request_id") or telemetry.request_id_var.get()
    cache_key = cache.make_key(request, prompt_version)
    fresh = request.metadata.get("fresh", False)

    if not fresh:
        cached_result = cache.get(cache_key)
        if cached_result is not None:
            on_chunk(cached_result.text)
            log.info("llm.attempt", extra={
                "draft_id": draft_id,
                "request_id": req_id,
                "client": client_id,
                "attempt": 1,
                "provider": cached_result.provider,
                "model": cached_result.model,
                "prompt_version": prompt_version,
                "wait_queue_ms": 0.0,
                "first_token_ms": 0.0,
                "duration_ms": 0.0,
                "status": "done",
                "kind": None,
                "fallback": False,
                "tokens": cached_result.usage,
                "cost": 0.0,
                "cost_estimated": False,
                "cached": True,
            })
            return cached_result
    else:
        cache.record_bypass()

    queue_start = time.perf_counter()
    _semaphore.acquire()
    queue_wait_ms = round((time.perf_counter() - queue_start) * 1000, 1)

    try:
        current_provider_name = os.getenv("LLM_PROVIDER", PROVIDER)
        current_fallback_name = os.getenv("LLM_FALLBACK", FALLBACK)

        providers_to_try = [(current_provider_name, False)]
        if current_fallback_name and current_fallback_name != "none":
            providers_to_try.append((current_fallback_name, True))

        total_deadline = time.perf_counter() + TOTAL_TIMEOUT
        last_error = None

        for prov_name, is_fallback in providers_to_try:
            prov = _create_provider(prov_name)
            max_tries = 1 if (is_fallback or prov_name == "mock") else MAX_ATTEMPTS

            for attempt in range(1, max_tries + 1):
                if is_cancelled():
                    raise ProviderError(kind="cancelled", message="Генерацію скасовано")

                try:
                    budget.check()
                except budget.BudgetExceeded as be:
                    raise ProviderError(kind="budget", message=str(be), retryable=False) from be

                attempt_start = time.perf_counter()
                first_token_ms = None
                text_parts = []
                final_usage = None
                final_finish_reason = None

                try:
                    for chunk in prov.stream(request):
                        if is_cancelled():
                            raise ProviderError(kind="cancelled", message="Генерацію скасовано")
                        if chunk.text:
                            if first_token_ms is None:
                                first_token_ms = round((time.perf_counter() - attempt_start) * 1000, 1)
                            text_parts.append(chunk.text)
                            on_chunk(chunk.text)
                        if chunk.usage:
                            final_usage = chunk.usage
                        if chunk.finish_reason:
                            final_finish_reason = chunk.finish_reason

                    duration_ms = round((time.perf_counter() - attempt_start) * 1000, 1)
                    full_text = "".join(text_parts)
                    cost, is_est = costs.estimate(final_usage, provider=prov.name, model=prov.model, text=full_text)
                    budget.record(cost)

                    result = Result(
                        text=full_text,
                        provider=prov.name,
                        model=prov.model,
                        usage=final_usage or {},
                        finish_reason=final_finish_reason or "stop",
                        elapsed=round(duration_ms / 1000.0, 3),
                        cost=cost,
                        cost_estimated=is_est,
                        cached=False,
                    )

                    level = logging.WARNING if (attempt > 1 or is_fallback) else logging.INFO
                    log.log(level, "llm.attempt", extra={
                        "draft_id": draft_id,
                        "request_id": req_id,
                        "client": client_id,
                        "attempt": attempt,
                        "provider": prov.name,
                        "model": prov.model,
                        "prompt_version": prompt_version,
                        "wait_queue_ms": queue_wait_ms,
                        "first_token_ms": first_token_ms or duration_ms,
                        "duration_ms": duration_ms,
                        "status": "done",
                        "kind": None,
                        "fallback": is_fallback,
                        "tokens": final_usage or {},
                        "cost": cost,
                        "cost_estimated": is_est,
                        "cached": False,
                    })

                    cache.put(cache_key, result)
                    return result

                except ProviderError as pe:
                    duration_ms = round((time.perf_counter() - attempt_start) * 1000, 1)
                    last_error = pe

                    if pe.kind == "budget":
                        log.error("llm.attempt", extra={
                            "draft_id": draft_id,
                            "request_id": req_id,
                            "client": client_id,
                            "attempt": attempt,
                            "provider": prov.name,
                            "model": prov.model,
                            "prompt_version": prompt_version,
                            "wait_queue_ms": queue_wait_ms,
                            "first_token_ms": first_token_ms,
                            "duration_ms": duration_ms,
                            "status": "failed",
                            "kind": "budget",
                            "fallback": is_fallback,
                            "tokens": final_usage or {},
                            "cost": 0.0,
                            "cost_estimated": False,
                            "cached": False,
                        })
                        raise pe

                    partial_text = "".join(text_parts)
                    cost, is_est = costs.estimate(final_usage, provider=prov.name, model=prov.model, text=partial_text)
                    if cost:
                        budget.record(cost)

                    is_final = (attempt == max_tries and (is_fallback or current_fallback_name == "none"))
                    err_level = logging.ERROR if is_final else logging.WARNING
                    log.log(err_level, "llm.attempt", extra={
                        "draft_id": draft_id,
                        "request_id": req_id,
                        "client": client_id,
                        "attempt": attempt,
                        "provider": prov.name,
                        "model": prov.model,
                        "prompt_version": prompt_version,
                        "wait_queue_ms": queue_wait_ms,
                        "first_token_ms": first_token_ms,
                        "duration_ms": duration_ms,
                        "status": "failed",
                        "kind": pe.kind,
                        "fallback": is_fallback,
                        "tokens": final_usage or {},
                        "cost": cost,
                        "cost_estimated": is_est,
                        "cached": False,
                    })

                    if not pe.retryable or attempt >= max_tries or time.perf_counter() >= total_deadline:
                        break

                    sleep_dur = min(BACKOFF_MAX, BACKOFF_BASE * (2 ** (attempt - 1)) + random.uniform(0, 0.1))
                    time.sleep(sleep_dur)

        if last_error:
            raise last_error
        raise ProviderError(kind="unknown", message="Не вдалося отримати відповідь після всіх спроб", retryable=False)

    finally:
        _semaphore.release()
