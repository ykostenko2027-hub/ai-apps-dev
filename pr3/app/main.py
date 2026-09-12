from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from app.llm import (
    ask,
    LLMError,
    LLMAuthenticationError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUnavailableError
)

app = FastAPI(title="Помічник служби підтримки — ПР3")

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
INDEX_FILE = TEMPLATES_DIR / "index.html"


class AskRequest(BaseModel):
    question: Optional[str] = Field(None, description="Текст запитання клієнта")
    text: Optional[str] = Field(None, description="Альтернативне поле для запиту")


class AskResponse(BaseModel):
    answer: str
    model: str
    elapsed: float
    tokens: Optional[dict] = None


@app.get("/")
async def get_index():
    """Віддає веб-сторінку помічника служби підтримки."""
    if not INDEX_FILE.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Файл шаблону index.html не знайдено."
        )
    return FileResponse(INDEX_FILE, media_type="text/html")


@app.post("/api/ask", response_model=AskResponse)
async def api_ask(req: AskRequest):
    """Обробляє запитання клієнта та повертає відповідь на основі правил магазину."""
    query = (req.question or req.text or "").strip()
    if not query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Будь ласка, введіть текст звернення. Запит не може бути порожнім."
        )

    try:
        result = await ask(query)
        return AskResponse(
            answer=result["answer"],
            model=result["model"],
            elapsed=result["elapsed"],
            tokens=result.get("tokens")
        )
    except LLMAuthenticationError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Помилка автентифікації: недійсний API-ключ або відсутній доступ до моделі."
        ) from exc
    except LLMRateLimitError as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Перевищено ліміт запитів до моделі (Rate Limit). Зачекайте хвилину і спробуйте знову."
        ) from exc
    except LLMTimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Час очікування відповіді від сервісу моделі вичерпано (Timeout). Спробуйте ще раз."
        ) from exc
    except LLMUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Сервіс мовної моделі наразі недоступний. Спробуйте пізніше."
        ) from exc
    except LLMError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Помилка при зверненні до моделі: {exc.message}"
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Внутрішня помилка сервера при обробці запиту."
        ) from exc


# Аліас для сумісності з попереднім ендпоінтом /chat
@app.post("/chat")
async def chat_endpoint(req: AskRequest):
    return await api_ask(req)