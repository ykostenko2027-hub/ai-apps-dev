"""Контракт відповіді помічника: що саме модель має повернути і як це перевіряється."""

import json
import re
from typing import Literal, Optional
from pydantic import BaseModel, Field, ValidationError

TopicType = Literal[
    "замовлення",
    "доставка",
    "оплата",
    "повернення",
    "гарантія",
    "підтримка",
    "інше",
]


class SupportResponse(BaseModel):
    """Схема структурованої відповіді помічника служби підтримки."""

    reply: str = Field(
        description="Текст ввічливої відповіді для клієнта українською мовою."
    )
    topic: TopicType = Field(
        description="Тема звернення з фіксованого переліку категорій."
    )
    based_on_rules: bool = Field(
        description="true, якщо відповідь повністю ґрунтується на наданих правилах; false, якщо правил недостатньо або питання поза правилами."
    )
    needs_clarification: bool = Field(
        description="true, якщо звернення неповне/неоднозначне і в клієнта потрібно щось уточнити."
    )
    escalate_to_operator: bool = Field(
        description="true, якщо розмову необхідно передати людині-оператору."
    )
    order_number: Optional[str] = Field(
        default=None,
        description="Шестизначний номер замовлення (тільки 6 цифр), якщо клієнт його назвав, інакше null.",
    )


def output_schema() -> dict:
    """Повернути JSON Schema відповіді помічника."""
    return SupportResponse.model_json_schema()


def validate(raw: str) -> dict:
    """Перевірити сиру відповідь моделі й повернути дані, яким можна довіряти структурно.

    Якщо текст не є JSON або не відповідає схемі, викидає ValueError з детальним описом.
    """
    cleaned = raw.strip()
    # Якщо модель випадково обгорнула відповідь у markdown ```json ... ```
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Відповідь моделі не є валідним JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Очікувався JSON-обʼєкт (словник), отримано: {type(data).__name__}")

    try:
        model_obj = SupportResponse.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"Помилка валідації схеми відповіді: {exc}") from exc

    res = model_obj.model_dump()

    # Програмна перевірка формату номера замовлення (має бути рівно 6 цифр)
    if res.get("order_number"):
        num_str = str(res["order_number"]).strip()
        match = re.search(r"\b\d{6}\b", num_str)
        if match:
            res["order_number"] = match.group(0)
        else:
            # Якщо модель вигадала некоректний номер або рядок
            res["order_number"] = None

    return res

