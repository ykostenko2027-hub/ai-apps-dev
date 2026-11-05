"""Схема розбору звернення клієнта та валідація результату.

Контракт між моделлю на кроці розбору та подальшим workflow.
"""

import copy
import json
from typing import Any

import jsonschema

REQUEST_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "kind": {
            "type": "string",
            "enum": ["return", "cancel", "status", "other"],
            "description": "Вид звернення: return (повернення), cancel (скасування), status (стан замовлення), other (інше)",
        },
        "order_id": {
            "type": ["string", "null"],
            "description": "Номер замовлення, якщо клієнт назвав його явно у тексті; якщо не назвав — обов'язково null",
        },
        "item": {
            "type": ["string", "null"],
            "description": "Назва товару або артикул зі звернення; якщо не названо — null",
        },
        "reason": {
            "type": ["string", "null"],
            "enum": ["not_suitable", "defect", "wrong_item", "damaged", None],
            "description": "Причина повернення, якщо вказана; якщо не вказана або для іншого виду — null",
        },
        "multiple_actions": {
            "type": "boolean",
            "description": "True, якщо в тексті міститься більше однієї дії/прохання (наприклад, скасування і повернення)",
        },
    },
    "required": ["kind", "order_id", "item", "reason", "multiple_actions"],
    "additionalProperties": False,
}


def request_schema() -> dict:
    """Повернути JSON Schema результату розбору звернення."""
    return copy.deepcopy(REQUEST_SCHEMA)


def validate(raw: str) -> dict:
    """Перевірити сиру відповідь моделі на кроці розбору й повернути словник."""
    try:
        data = json.loads(raw)
    except Exception as exc:
        raise ValueError(f"Відповідь моделі не є валідним JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Очікувався JSON-об'єкт, отримано: {type(data).__name__}")

    try:
        jsonschema.validate(instance=data, schema=REQUEST_SCHEMA)
    except jsonschema.ValidationError as exc:
        raise ValueError(f"Помилка валідації схеми: {exc.message}") from exc

    return data
