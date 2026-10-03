import os
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv

load_dotenv()

from app.rag import RAGPipeline

app = FastAPI(title="RAG Помічник — Магазин «Сузірʼя»")
templates = Jinja2Templates(directory="app/templates")

pipeline = RAGPipeline(
    search_top_k=8,
    model_top_k=4,
    similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.35")),
    model_name=os.getenv("MODEL_NAME", "gpt-4o-mini"),
    token_budget=1500
)

@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

@app.post("/ask", response_class=HTMLResponse)
async def ask_question(request: Request, question: str = Form(...)):
    q = question.strip()
    if not q:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "result": None, "error": "Питання не може бути порожнім."}
        )

    try:
        response_data = pipeline.run(q)
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "result": response_data, "question": q}
        )
    except Exception as e:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "result": None, "error": f"Сталася помилка: {str(e)}"}
        )

@app.post("/api/ask")
async def api_ask(payload: dict):
    question = payload.get("question", "").strip()
    if not question:
        return JSONResponse(status_code=400, content={"error": "Question is empty"})
    
    result = pipeline.run(question)
    return result.model_dump()