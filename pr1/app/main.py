from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse

from . import weather

app = FastAPI(title="Погода — ПР1")

INDEX_PAGE = Path(__file__).parent / "templates" / "index.html"


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_PAGE.read_text(encoding="utf-8")


@app.get("/api/weather")
def api_weather(city: str = Query("", description="Назва міста")):
    if not city or not city.strip():
        raise HTTPException(status_code=400, detail="Назва міста не може бути порожньою")
    try:
        return weather.get_current_weather(city)
    except weather.WeatherError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception:
        raise HTTPException(status_code=500, detail="Внутрішня помилка сервера")
