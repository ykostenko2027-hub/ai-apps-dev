"""Модуль роботи з мовною моделлю: єдине місце застосунку, яке знає про API."""

import logging
import os
import time
from typing import Optional

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

from . import schema

load_dotenv()

logger = logging.getLogger(__name__)

# Доступ до сервісу.
BASE_URL = os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
API_KEY = os.getenv("LLM_API_KEY", "")
MODEL = os.getenv("LLM_MODEL", "gemini-3.5-flash-lite")

# Параметри генерації.
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1500"))
TIMEOUT = float(os.getenv("LLM_TIMEOUT", "30"))

# Бюджет токенів.
TOKEN_BUDGET = int(os.getenv("LLM_TOKEN_BUDGET", "3000"))

# Клієнт як singleton
_client: Optional[OpenAI] = None


class LLMError(Exception):
    """Помилка роботи з моделлю, зрозуміла веб-рівню."""

    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


SYSTEM_INSTRUCTION = """Ти — ввічливий і компетентний асистент служби підтримки інтернет-магазину «Сузірʼя».

Твоя задача — відповідати на звернення клієнтів і повертати результат ВИКЛЮЧНО у вигляді структурованого JSON-обʼєкта відповідно до наданої схеми.

ОБМЕЖЕННЯ ТА ПРАВИЛА:
1. Мова відповідей — виключно українська. Стиль — ввічливий, професійний, лаконічний.
2. Відповідай ВИКЛЮЧНО на підставі наданих Офіційних правил обслуговування магазину. Категорично заборонено вигадувати правила, терміни, суми, умови доставки або повернення, якщо їх немає в тексті правил.
3. Якщо питання клієнта не описане в правилах або виходить за їхні межі: чесно повідоми про це клієнта, встанови based_on_rules=false та запропонуй передати розмову оператору (escalate_to_operator=true).
4. Якщо звернення неповне або неоднозначне (наприклад, клієнт просить повернення без зазначення причини, моделі чи стану товару): ввічливо постав уточнююче запитання та встанови needs_clarification=true.
5. Замовлення: зверни особливу увагу на номер замовлення. Він складається з 6 цифр. Якщо клієнт назвав номер замовлення в поточному зверненні або він міститься в попередніх повідомленнях історії — зафіксуй його в order_number і НЕ перепитуй знову! Якщо номера немає — встанови order_number=null.
6. Тема звернення (topic) обирається суворо з фіксованого списку: "замовлення", "доставка", "оплата", "повернення", "гарантія", "підтримка", "інше".

ПРИКЛАДИ МЕЖОВИХ ВИПАДКІВ:

Приклад 1 (відповідь є в правилах):
Звернення: "Замовлення на 1500 грн. Скільки коштуватиме доставка?"
Результат:
{"reply": "Доставка замовлення на суму 1500 грн оплачується покупцем за тарифами перевізника, оскільки безкоштовна доставка надається від 2000 грн.", "topic": "доставка", "based_on_rules": true, "needs_clarification": false, "escalate_to_operator": false, "order_number": null}

Приклад 2 (відповіді немає в правилах / питання поза правилами):
Звернення: "Чи доставляєте ви до Польщі і скільки це коштує?"
Результат:
{"reply": "На жаль, за нашими правилами доставка здійснюється тільки по Україні, а міжнародна доставка не передбачена. Можу передати ваше звернення оператору для уточнення деталей.", "topic": "доставка", "based_on_rules": false, "needs_clarification": false, "escalate_to_operator": true, "order_number": null}

Приклад 3 (неоднозначне звернення / потрібне уточнення):
Звернення: "Замовив навушники три дні тому. Хочу повернути. Що робити?"
Результат:
{"reply": "Уточніть, будь ласка: товар належної якості чи має дефект, і яка це модель навушників? Зверніть увагу: навушники-вкладиші належної якості поверненню не підлягають. Якщо ж товар з дефектом, діє гарантійне обслуговування.", "topic": "повернення", "based_on_rules": true, "needs_clarification": true, "escalate_to_operator": false, "order_number": null}
"""


def get_client() -> OpenAI:
    """Повернути готовий до роботи клієнт сервісу (singleton)."""
    global _client
    if _client is None:
        if not API_KEY:
            raise LLMError("LLM_API_KEY не задано в .env", status_code=500)
        _client = OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=TIMEOUT)
    return _client


def estimate_tokens(text: str) -> int:
    """Оцінити, скільки токенів займе текст.

    Для української мови один токен відповідає приблизно 2.5 символам.
    """
    if not text:
        return 0
    return max(1, int(len(text) / 2.5))


def fit_budget(history: list[dict], budget: int) -> list[dict]:
    """Повернути частину історії, що вміщується в бюджет.

    Відкидаються найстаріші повідомлення, щоб зберегти найактуальніший контекст.
    """
    if not history or budget <= 0:
        return []

    fitted = list(history)
    current_tokens = sum(estimate_tokens(turn.get("content", "")) for turn in fitted)

    while fitted and current_tokens > budget:
        removed = fitted.pop(0)
        current_tokens -= estimate_tokens(removed.get("content", ""))

    return fitted


def build_messages(message: str, history: list[dict], context: str) -> list[dict]:
    """Скласти список повідомлень для моделі.

    Частини запиту лишаються чітко розмежованими:
    1. Системна інструкція з роллю, обмеженнями та прикладами.
    2. Офіційні правила обслуговування (контекст).
    3. Історія повідомлень (скорочена під бюджет).
    4. Поточне повідомлення користувача.
    """
    mandatory_tokens = (
        estimate_tokens(SYSTEM_INSTRUCTION)
        + estimate_tokens(context)
        + estimate_tokens(message)
        + MAX_TOKENS
    )
    history_budget = max(0, TOKEN_BUDGET - mandatory_tokens)
    fitted_history = fit_budget(history, history_budget)

    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "system", "content": f"ОФІЦІЙНІ ПРАВИЛА ОБСЛУГОВУВАННЯ «СУЗІРʼЯ»:\n{context}"},
    ]
    for turn in fitted_history:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": message})
    return messages


def ask(message: str, history: list[dict], context: str) -> dict:
    """Поставити моделі питання й повернути перевірений результат."""
    client = get_client()
    messages = build_messages(message, history, context)
    json_schema = schema.output_schema()

    started = time.perf_counter()
    raw_content = ""
    usage_info = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    # Виклик моделі з обробкою мережевих збоїв та таймаутів
    try:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "SupportResponse", "schema": json_schema},
            },
        )
        elapsed = time.perf_counter() - started
        raw_content = completion.choices[0].message.content or ""
        if completion.usage:
            usage_info = {
                "prompt_tokens": completion.usage.prompt_tokens,
                "completion_tokens": completion.usage.completion_tokens,
                "total_tokens": completion.usage.total_tokens,
            }
    except AuthenticationError as exc:
        logger.error("LLM Auth error: %s", exc)
        raise LLMError("Помилка автентифікації API ключа.", status_code=401) from exc
    except RateLimitError as exc:
        logger.error("LLM Rate limit: %s", exc)
        raise LLMError("Перевищено ліміт запитів до моделі. Спробуйте пізніше.", status_code=429) from exc
    except APITimeoutError as exc:
        logger.error("LLM Timeout: %s", exc)
        raise LLMError("Час очікування відповіді від моделі вичерпано.", status_code=504) from exc
    except (APIConnectionError, InternalServerError) as exc:
        logger.error("LLM Connection/Server error: %s", exc)
        raise LLMError("Сервіс моделі тимчасово недоступний.", status_code=503) from exc
    except Exception as exc:
        logger.error("Unexpected LLM error: %s", exc)
        raise LLMError(f"Помилка виклику моделі: {exc}", status_code=500) from exc

    # Перевірка сирої відповіді за схемою
    try:
        validated_result = schema.validate(raw_content)
    except ValueError as val_err:
        logger.warning("Валідація схеми не вдалася (%s). Сира відповідь: %s", val_err, raw_content)
        # Спроба одноразового повторного запиту з текстом помилки
        try:
            retry_messages = list(messages)
            retry_messages.append({"role": "assistant", "content": raw_content})
            retry_messages.append({
                "role": "user",
                "content": f"Твоя попередня відповідь містить помилку схеми: {val_err}. Виправ це і надай коректний JSON-обʼєкт.",
            })
            retry_completion = client.chat.completions.create(
                model=MODEL,
                messages=retry_messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "SupportResponse", "schema": json_schema},
                },
            )
            retry_raw = retry_completion.choices[0].message.content or ""
            validated_result = schema.validate(retry_raw)
            if retry_completion.usage:
                usage_info["prompt_tokens"] += retry_completion.usage.prompt_tokens
                usage_info["completion_tokens"] += retry_completion.usage.completion_tokens
                usage_info["total_tokens"] += retry_completion.usage.total_tokens
        except Exception:
            # Якщо й повтор не вдався — піднімаємо зрозумілий збій невалідної відповіді
            raise LLMError("Модель повернула невалідну відповідь, яку не вдалося розібрати за схемою.", status_code=502)

    # Якщо модель не вказала order_number у поточному кроці, шукаємо в історії повідомлень клієнта
    if not validated_result.get("order_number"):
        import re
        for turn in reversed(history):
            if turn.get("role") == "user":
                match = re.search(r"\b\d{6}\b", turn.get("content", ""))
                if match:
                    validated_result["order_number"] = match.group(0)
                    break

    return {
        "result": validated_result,
        "model": MODEL,
        "elapsed": round(elapsed, 2),
        "usage": usage_info,
    }

