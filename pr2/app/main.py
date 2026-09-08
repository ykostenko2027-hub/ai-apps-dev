"""Веб-рівень застосунку: сторінка із завантаженням файлу і JSON-ендпоінт.

Цей файл не має знати про `ultralytics`, ваги моделі й формат її «сирого»
виводу — усе це лишається в `app/detector.py`. Тут вирішується інше: що
застосунок віддає клієнтові та з яким HTTP-статусом.

Запуск із папки pr2:

    uvicorn app.main:app --reload

Далі відкрийте http://127.0.0.1:8000
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import HTMLResponse

from . import detector


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        detector.load_model()
    except Exception:
        pass
    yield


app = FastAPI(title="Детекція обʼєктів — ПР2", lifespan=lifespan)

INDEX_PAGE = Path(__file__).parent / "templates" / "index.html"


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Віддати сторінку із завантаженням зображення."""
    return INDEX_PAGE.read_text(encoding="utf-8")


@app.post("/api/detect")
async def api_detect(
    image: UploadFile = File(...),
    confidence: float = Form(default=detector.DEFAULT_CONFIDENCE, ge=0.0, le=1.0),
):
    """Повернути знайдені на зображенні обʼєкти у форматі JSON.

    Зараз виняток із модуля inference не обробляється — застосунок просто
    впаде з помилкою 500. Спроєктуйте обробку самі: які збої можливі
    (не зображення, порожній файл, помилка моделі), який HTTP-статус
    відповідає кожному з них і що в такому разі отримає клієнт.

    Поріг упевненості поки що жорстко зашитий у модулі. Вирішіть, чи має
    користувач змогу його змінювати, і якщо так — як передати це сюди.
    """
    if not image or not image.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Файл не вибрано або ім'я файлу відсутнє."
        )

    content = await image.read()
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Отримано порожній файл. Будь ласка, виберіть коректне зображення."
        )

    if image.content_type and not (
        image.content_type.startswith("image/") or image.content_type == "application/octet-stream"
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Непідтримуваний тип файлу ({image.content_type}). Очікується файл зображення."
        )

    try:
        result = detector.detect(content, confidence=confidence)
        return result
    except detector.InvalidImageError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc)
        ) from exc
    except detector.ModelInferenceError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Помилка виконання інференсу моделі: {exc}"
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Несподівана помилка сервера: {exc}"
        ) from exc
