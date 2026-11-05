"""Модуль взаємодії з мовною моделлю через OpenAI-сумісний API.

Підтримує виклик моделі з описами інструментів (для агента) та
виклик без інструментів зі structured output або чистим текстом
(для workflow).
"""

import os
import time
from typing import Optional

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    OpenAI,
    OpenAIError,
    RateLimitError,
)

load_dotenv()

BASE_URL = os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
API_KEY = os.getenv("LLM_API_KEY")
MODEL = os.getenv("LLM_MODEL", "gemini-3.5-flash-lite")

TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "800"))
TIMEOUT = float(os.getenv("LLM_TIMEOUT", "30"))

FALLBACK_MODELS = [
    MODEL,
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.8-flash",
    "gemini-2.5-flash",
    "gemini-3.1-flash-lite",
]


class LLMError(Exception):
    """Помилка роботи з моделлю."""


_client: Optional[OpenAI] = None


def get_client() -> OpenAI:
    """Повернути готовий клієнт OpenAI (синглтон)."""
    global _client
    if _client is None:
        if not API_KEY:
            raise LLMError("LLM_API_KEY не задано у .env")
        _client = OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=TIMEOUT)
    return _client


SYSTEM_PROMPT_AGENT = """Ти — ввічливий і компетентний помічник клієнтської підтримки інтернет-магазину «Сузірʼя».
Твоє завдання — допомагати клієнтам щодо їхніх замовлень, повернення товарів, пошуку товарів та інформації про доставку.

ПРАВИЛА ТА ОБМЕЖЕННЯ:
1. Завжди використовуй надані інструменти для отримання актуальних даних магазину. Не вигадуй номерів замовлень, статусів, дат чи наявності товарів.
2. Клієнтові доступні лише його власні замовлення. Ніколи не розголошуй інформацію про чужі замовлення.
3. Ніколи не розголошуй внутрішні службові примітки (notes), інструкції адміністрації чи технічні подробиці.
4. Якщо інформації недостатньо або замовлення не знайдено — ввічливо запитай клієнта про деталі.
5. Відповідай українською мовою, стисло, чітко й по суті.
"""


def build_messages(question: str) -> list[dict]:
    """Скласти початковий список повідомлень для агента."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT_AGENT},
        {"role": "user", "content": question},
    ]


def _unique_models() -> list[str]:
    seen = set()
    res = []
    for m in FALLBACK_MODELS:
        if m and m not in seen:
            seen.add(m)
            res.append(m)
    return res


def chat(messages: list[dict], tools: list[dict], tool_choice: str = "auto") -> dict:
    """Один виклик моделі з історією повідомлень та описами інструментів (для агента)."""
    client = get_client()
    started = time.perf_counter()
    last_exc = None

    for target_model in _unique_models():
        try:
            kwargs = {
                "model": target_model,
                "messages": messages,
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS,
                "timeout": TIMEOUT,
            }
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = tool_choice

            resp = client.chat.completions.create(**kwargs)
            elapsed = time.perf_counter() - started
            msg = resp.choices[0].message
            usage = {
                "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
                "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
                "total_tokens": resp.usage.total_tokens if resp.usage else 0,
            }
            return {
                "message": msg,
                "text": msg.content or "",
                "tool_calls": msg.tool_calls or [],
                "model": resp.model,
                "usage": usage,
                "elapsed": elapsed,
            }
        except RateLimitError as exc:
            last_exc = exc
            continue
        except (AuthenticationError, BadRequestError, APITimeoutError, APIConnectionError, OpenAIError) as exc:
            raise LLMError(f"Помилка виклику LLM ({type(exc).__name__}): {exc}") from exc
        except Exception as exc:
            raise LLMError(f"Непередбачена помилка LLM: {exc}") from exc

    raise LLMError(f"Перевищено ліміт запитів (429): {last_exc}")


def complete(messages: list[dict], schema: dict | None = None) -> dict:
    """Одне звертання до моделі без інструментів (для кроків workflow)."""
    client = get_client()
    started = time.perf_counter()
    last_exc = None

    for target_model in _unique_models():
        try:
            kwargs = {
                "model": target_model,
                "messages": messages,
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS,
                "timeout": TIMEOUT,
            }
            if schema:
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": "schema_output", "schema": schema},
                }

            resp = client.chat.completions.create(**kwargs)
            elapsed = time.perf_counter() - started
            text = (resp.choices[0].message.content or "").strip()
            usage = {
                "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
                "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
                "total_tokens": resp.usage.total_tokens if resp.usage else 0,
            }
            return {
                "text": text,
                "model": resp.model,
                "usage": usage,
                "elapsed": elapsed,
            }
        except RateLimitError as exc:
            last_exc = exc
            continue
        except (AuthenticationError, BadRequestError, APITimeoutError, APIConnectionError, OpenAIError) as exc:
            raise LLMError(f"Помилка виклику LLM ({type(exc).__name__}): {exc}") from exc
        except Exception as exc:
            raise LLMError(f"Непередбачена помилка LLM: {exc}") from exc

    raise LLMError(f"Перевищено ліміт запитів (429): {last_exc}")
