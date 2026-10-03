from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from . import extraction
from .images import ImageError
from .llm import LLMError

app = FastAPI(title="Розбір рахунків — ПР7")

INDEX_PAGE = Path(__file__).parent / "templates" / "index.html"
SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}

app.mount("/samples", StaticFiles(directory=SAMPLES_DIR), name="samples")


def result_to_dict(result: extraction.Result) -> dict:
    return {
        "decision": result.decision,
        "reasons": result.reasons,
        "document": result.document,
        "issues": [asdict(issue) for issue in result.issues],
        "image": result.image,
        "model": result.model,
        "elapsed": result.elapsed,
        "usage": result.usage,
    }


@app.get("/", response_class=HTMLResponse)
def page() -> str:
    return INDEX_PAGE.read_text(encoding="utf-8")


@app.get("/api/samples")
def api_samples() -> dict:
    groups = {}
    for folder in sorted(p for p in SAMPLES_DIR.iterdir() if p.is_dir()):
        files = sorted(f.name for f in folder.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES)
        if files:
            groups[folder.name] = files
    return groups


@app.post("/api/extract")
async def api_extract(image: UploadFile = File(...)) -> dict:
    if not image.filename:
        raise HTTPException(status_code=400, detail="Файл не вибрано.")
    suffix = Path(image.filename).suffix.lower()
    if suffix and suffix not in IMAGE_SUFFIXES:
        allowed = ", ".join(sorted(IMAGE_SUFFIXES))
        raise HTTPException(status_code=400, detail=f"Непідтримуваний формат файлу: {suffix}. Дозволені формати: {allowed}")
    try:
        content = await image.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Не вдалося прочитати файл: {exc}")
    if not content:
        raise HTTPException(status_code=400, detail="Надісланий файл порожній.")
    try:
        result = extraction.process(content)
        return result_to_dict(result)
    except ImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Внутрішня помилка обробки: {exc}")
