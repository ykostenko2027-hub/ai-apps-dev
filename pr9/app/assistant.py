"""Агентний підхід із ПР8: цикл помічника клієнтської підтримки.

Модель сама обирає, які інструменти викликати й у якому порядку.
Код лише валідує виклики через tools.call та обмежує кількість раундів (TOOL_MAX_ROUNDS).
"""

import json
import os
import time
from dataclasses import dataclass, field

from dotenv import load_dotenv

from . import llm, tools

load_dotenv()

MAX_ROUNDS = int(os.getenv("TOOL_MAX_ROUNDS", "3"))


@dataclass
class ToolTrace:
    """Запис про один виклик інструмента."""
    round: int
    name: str
    arguments: str
    status: str
    reason: str | None = None
    result: dict | list | str | None = None
    elapsed: float | None = None


@dataclass
class Answer:
    """Підсумкова відповідь агента."""
    text: str
    calls: list[ToolTrace] = field(default_factory=list)
    rounds: int = 0
    stopped: str = "answer"
    model: str | None = None
    elapsed: dict = field(default_factory=dict)
    usage: dict | None = None


def answer(question: str, customer_id: str) -> Answer:
    """Обробити запитання клієнта агентним циклом."""
    messages = llm.build_messages(question)
    ctx = tools.Context(customer_id=customer_id)

    traces: list[ToolTrace] = []
    rounds = 0
    total_model_time = 0.0
    total_tools_time = 0.0
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    last_model = None
    last_text = ""

    while rounds < MAX_ROUNDS:
        rounds += 1
        resp = llm.chat(messages, tools.specs())
        last_model = resp.get("model")
        total_model_time += resp.get("elapsed", 0.0)

        usage = resp.get("usage", {})
        total_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
        total_usage["completion_tokens"] += usage.get("completion_tokens", 0)
        total_usage["total_tokens"] += usage.get("total_tokens", 0)

        tool_calls = resp.get("tool_calls", [])
        text = resp.get("text", "")
        if text:
            last_text = text

        if not tool_calls:
            return Answer(
                text=text,
                calls=traces,
                rounds=rounds,
                stopped="answer",
                model=last_model,
                elapsed={"model": round(total_model_time, 3), "tools": round(total_tools_time, 3)},
                usage=total_usage,
            )

        # Додаємо повідомлення моделі з tool_calls
        messages.append(resp["message"])

        # Виконуємо кожен виклик
        for tc in tool_calls:
            t0 = time.perf_counter()
            tool_res = tools.call(tc.function.name, tc.function.arguments, ctx)
            dt = time.perf_counter() - t0
            total_tools_time += dt

            traces.append(
                ToolTrace(
                    round=rounds,
                    name=tc.function.name,
                    arguments=tc.function.arguments,
                    status=tool_res.status,
                    reason=tool_res.reason,
                    result=tool_res.content,
                    elapsed=round(dt, 4),
                )
            )

            tool_content = tool_res.content if tool_res.content is not None else tool_res.reason
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(tool_content, ensure_ascii=False) if not isinstance(tool_content, str) else tool_content,
            })

    # Якщо досягнуто ліміту раундів
    final_text = last_text or "Вибачте, перевищено ліміт кроків для обробки вашого запиту."
    return Answer(
        text=final_text,
        calls=traces,
        rounds=rounds,
        stopped="limit",
        model=last_model,
        elapsed={"model": round(total_model_time, 3), "tools": round(total_tools_time, 3)},
        usage=total_usage,
    )
