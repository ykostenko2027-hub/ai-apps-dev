import os
import queue
import random
import threading
import time
from typing import Callable

from dotenv import load_dotenv

from .providers.base import Provider, ProviderError, Request, Result
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

MODEL = os.getenv("LLM_MODEL")
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "600"))

_semaphore = threading.Semaphore(MAX_CONCURRENT)


def settings() -> dict:
    return {
        "provider": PROVIDER,
        "fallback": FALLBACK,
        "model": MODEL,
        "timeout": TIMEOUT,
        "idle_timeout": IDLE_TIMEOUT,
        "total_timeout": TOTAL_TIMEOUT,
        "max_attempts": MAX_ATTEMPTS,
        "max_concurrent": MAX_CONCURRENT,
    }


def make_provider(name: str) -> Provider:
    if name == "mock":
        return MockProvider()
    base_url = os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
    api_key = os.getenv("LLM_API_KEY", "")
    target_model = MODEL or "gemini-2.5-flash"
    if name != "openai":
        target_model = name
    return OpenAICompatProvider(base_url, api_key, target_model, TIMEOUT)


def _execute_stream(provider: Provider, request: Request, on_delta: Callable[[str], None],
                    is_cancelled: Callable[[], bool], start_time: float) -> Result:
    q: queue.Queue = queue.Queue()
    sentinel = object()

    def worker() -> None:
        try:
            for chunk in provider.stream(request):
                q.put((True, chunk))
            q.put((True, sentinel))
        except Exception as exc:
            q.put((False, exc))

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    accumulated = []
    last_usage = None
    finish_reason = "stop"

    while True:
        if is_cancelled():
            raise ProviderError("cancelled", "Операцію скасовано користувачем", retryable=False)
        remaining = TOTAL_TIMEOUT - (time.perf_counter() - start_time)
        if remaining <= 0:
            raise ProviderError("timeout", "Вичерпано загальний ліміт часу", retryable=False)
        step_timeout = min(IDLE_TIMEOUT, remaining)
        try:
            ok, item = q.get(timeout=step_timeout)
        except queue.Empty:
            raise ProviderError("timeout", "Таймаут бездіяльності в режимі streaming", retryable=True)

        if not ok:
            if isinstance(item, ProviderError):
                raise item
            raise ProviderError("unknown", str(item), detail=str(item))

        if item is sentinel:
            break

        if item.text:
            accumulated.append(item.text)
            on_delta(item.text)
        if item.usage:
            last_usage = item.usage
        if item.finish_reason:
            finish_reason = item.finish_reason

    full_text = "".join(accumulated)
    return Result(
        text=full_text,
        provider=provider.name,
        model=getattr(provider, "model", MODEL or "unknown"),
        usage=last_usage,
        finish_reason=finish_reason,
        elapsed=round(time.perf_counter() - start_time, 2),
    )


def generate(request: Request, on_delta: Callable[[str], None],
             on_status: Callable[[dict], None], is_cancelled: Callable[[], bool]) -> Result:
    acquired = _semaphore.acquire(timeout=TOTAL_TIMEOUT)
    if not acquired:
        raise ProviderError("rate_limit", "Черга запитів переповнена, повторіть пізніше", retryable=True, retry_after=2.0)

    try:
        start_time = time.perf_counter()
        active_provider_name = PROVIDER
        is_fallback_mode = False
        last_error = None

        while True:
            provider = make_provider(active_provider_name)
            for attempt in range(1, MAX_ATTEMPTS + 1):
                if is_cancelled():
                    raise ProviderError("cancelled", "Операцію скасовано користувачем", retryable=False)
                if (time.perf_counter() - start_time) >= TOTAL_TIMEOUT:
                    raise ProviderError("timeout", "Вичерпано загальний ліміт часу виконання", retryable=False)

                on_status({
                    "status": "running",
                    "attempt": attempt,
                    "provider": active_provider_name,
                    "fallback": is_fallback_mode,
                })

                try:
                    return _execute_stream(provider, request, on_delta, is_cancelled, start_time)
                except ProviderError as err:
                    last_error = err
                    if err.kind == "cancelled":
                        raise err
                    if not err.retryable or attempt >= MAX_ATTEMPTS:
                        break

                    delay = err.retry_after if err.retry_after is not None else min(
                        BACKOFF_MAX, BACKOFF_BASE * (2 ** (attempt - 1))
                    ) * random.uniform(0.5, 1.5)

                    if (time.perf_counter() - start_time) + delay >= TOTAL_TIMEOUT:
                        break

                    sleep_end = time.perf_counter() + delay
                    while time.perf_counter() < sleep_end:
                        if is_cancelled():
                            raise ProviderError("cancelled", "Операцію скасовано користувачем", retryable=False)
                        time.sleep(0.05)

            if not is_fallback_mode and FALLBACK != "none":
                is_fallback_mode = True
                active_provider_name = FALLBACK
                on_status({
                    "status": "running",
                    "attempt": 1,
                    "provider": active_provider_name,
                    "fallback": True,
                })
                continue

            if last_error:
                raise last_error
            raise ProviderError("unknown", "Не вдалося згенерувати відповідь", retryable=False)
    finally:
        _semaphore.release()
