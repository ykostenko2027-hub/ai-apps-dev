"""Веб-рівень застосунку: сторінка діалогу і JSON-ендпоінт.

Цей файл не має знати ані про провайдера моделі, ані про те, як
складається запит, ані про те, як виглядає схема відповіді, — усе це
лишається в `app/llm.py` і `app/schema.py`. Тут вирішується інше: що
застосунок приймає від сторінки, що віддає їй і з яким HTTP-статусом.

Запуск із папки pr4:

    uvicorn app.main:app --reload

Далі відкрийте http://127.0.0.1:8000
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from . import llm

app = FastAPI(title="Помічник служби підтримки — ПР4")

INDEX_PAGE = Path(__file__).parent / "templates" / "index.html"
CONTEXT_FILE = Path(__file__).parent.parent / "context.md"


class Turn(BaseModel):
    """Одна репліка розмови: `user` — клієнт, `assistant` — помічник."""

    role: str
    content: str


class ChatRequest(BaseModel):
    """Те, що надсилає сторінка: нове повідомлення й розмову до нього."""

    message: str
    history: list[Turn] = []


def load_context() -> str:
    """Прочитати правила організації, на підставі яких відповідає модель."""
    return CONTEXT_FILE.read_text(encoding="utf-8")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Віддати сторінку діалогу."""
    return INDEX_PAGE.read_text(encoding="utf-8")


@app.post("/api/chat")
def api_chat(payload: ChatRequest):
    """Повернути структуровану відповідь помічника у форматі JSON."""
    user_msg = payload.message.strip()
    if not user_msg:
        raise HTTPException(status_code=400, detail="Повідомлення не може бути порожнім.")

    history = [turn.model_dump() for turn in payload.history]

    try:
        context = load_context()
        return llm.ask(user_msg, history, context)
    except llm.LLMError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Внутрішня помилка сервера: {exc}")

