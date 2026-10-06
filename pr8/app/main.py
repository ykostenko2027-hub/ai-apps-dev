from dataclasses import asdict
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from shop import service
from . import assistant, tools, llm

app = FastAPI(title="Помічник клієнта — ПР8")

INDEX_PAGE = Path(__file__).parent / "templates" / "index.html"


class AskRequest(BaseModel):
    customer_id: str
    question: str


def answer_to_dict(result: assistant.Answer) -> dict:
    return {
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
    return INDEX_PAGE.read_text(encoding="utf-8")


@app.get("/api/customers")
def api_customers() -> list[dict]:
    return service.list_customers()


@app.get("/api/tools")
def api_tools() -> list[dict]:
    return tools.specs()


@app.post("/api/reset")
def api_reset() -> dict:
    service.reset()
    return {"ok": True}


@app.post("/api/ask")
def api_ask(payload: AskRequest) -> dict:
    clean_question = payload.question.strip()
    if not clean_question:
        raise HTTPException(status_code=400, detail="Питання не може бути порожнім.")

    known_ids = {c["customer_id"] for c in service.list_customers()}
    if payload.customer_id not in known_ids:
        raise HTTPException(status_code=404, detail=f"Клієнта з ідентифікатором '{payload.customer_id}' не знайдено.")

    try:
        result = assistant.answer(clean_question, payload.customer_id)
        return answer_to_dict(result)
    except llm.LLMError as exc:
        raise HTTPException(status_code=503, detail=f"Помилка зв'язку з мовною моделлю: {exc}")
    except service.Unavailable as exc:
        raise HTTPException(status_code=503, detail=f"Сервіс магазину тимчасово недоступний: {exc}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Внутрішня помилка обробки запиту: {exc}")
