"""Веб-рівень застосунку: сторінка звернень, черга оператора і JSON-ендпоінти.

FastAPI-ендпоінти для взаємодії з клієнтом і оператором.
Обробка збоїв: порожній текст, невідомий клієнт, відсутнє звернення,
помилки моделі та недоступність сервісу.
"""

from dataclasses import asdict
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from shop import service

from . import assistant, llm, store, workflow

app = FastAPI(title="Звернення клієнтів — ПР9")

INDEX_PAGE = Path(__file__).parent / "templates" / "index.html"


class RequestIn(BaseModel):
    """Нове звернення зі сторінки."""
    customer_id: str
    text: str
    mode: Literal["workflow", "agent"] = "workflow"


class DecisionIn(BaseModel):
    """Рішення оператора щодо дії, яка чекає підтвердження."""
    approve: bool
    comment: str = ""


def run_to_dict(run: workflow.Run) -> dict:
    """Перетворити стан звернення на те, що піде на сторінку."""
    data = asdict(run)
    data["mode"] = "workflow"
    data["limits"] = workflow.limits()
    return data


def answer_to_dict(result: assistant.Answer) -> dict:
    """Перетворити результат агента з ПР8 на те, що піде на сторінку."""
    return {
        "mode": "agent",
        "answer": result.text,
        "calls": [asdict(call) for call in result.calls],
        "rounds": result.rounds,
        "stopped": result.stopped,
        "model": result.model,
        "elapsed": result.elapsed,
        "usage": result.usage,
    }


@app.get("/", response_class=HTMLResponse)
def page() -> str:
    """Віддати HTML-сторінку."""
    return INDEX_PAGE.read_text(encoding="utf-8")


@app.get("/api/customers")
def api_customers() -> list[dict]:
    """Клієнти, від імені яких можна увійти на сторінці."""
    try:
        return service.list_customers()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Сервіс магазину недоступний: {exc}",
        ) from exc


@app.post("/api/reset")
def api_reset() -> dict:
    """Повернути дані магазину до початкового стану."""
    try:
        service.reset()
        return {"ok": True}
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Помилка скидання даних магазину: {exc}",
        ) from exc


@app.post("/api/requests")
def api_create(payload: RequestIn) -> dict:
    """Прийняти звернення й обробити його обраним способом."""
    # 1. Валідація тексту
    clean_text = payload.text.strip()
    if not clean_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Текст звернення не може бути порожнім.",
        )

    # 2. Валідація клієнта
    try:
        valid_customers = {c["customer_id"] for c in service.list_customers()}
    except Exception:
        valid_customers = set()

    if valid_customers and payload.customer_id not in valid_customers:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Клієнта з ідентифікатором '{payload.customer_id}' не знайдено.",
        )

    # 3. Виконання
    try:
        if payload.mode == "agent":
            res = assistant.answer(clean_text, payload.customer_id)
            return answer_to_dict(res)

        run = workflow.start(clean_text, payload.customer_id)
        return run_to_dict(run)

    except llm.LLMError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Помилка мовної моделі: {exc}",
        ) from exc
    except service.Unavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Сервіс магазину тимчасово недоступний.",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Внутрішня помилка обробки звернення: {exc}",
        ) from exc


@app.get("/api/runs")
def api_runs(status: str | None = None) -> list[dict]:
    """Звернення, оброблені workflow, — усі або з певним статусом."""
    return [run_to_dict(run) for run in store.list_runs(status)]


@app.get("/api/runs/{run_id}")
def api_run(run_id: str) -> dict:
    """Стан одного звернення."""
    run = store.load(run_id)
    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Звернення '{run_id}' не знайдено.",
        )
    return run_to_dict(run)


@app.post("/api/runs/{run_id}/decision")
def api_decision(run_id: str, payload: DecisionIn) -> dict:
    """Рішення оператора: підтвердити чи відхилити дію."""
    run = store.load(run_id)
    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Звернення '{run_id}' не знайдено.",
        )

    if run.status != "awaiting_approval" and run.decision is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Звернення '{run_id}' знаходиться у статусі '{run.status}' і не чекає підтвердження.",
        )

    try:
        updated = workflow.resume(run_id, payload.approve, payload.comment)
        return run_to_dict(updated)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Помилка застосування рішення оператора: {exc}",
        ) from exc
