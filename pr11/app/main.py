import json
import logging
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from . import budget, cache, costs, drafting, gateway, jobs, limits, samples, telemetry

telemetry.setup()
log = logging.getLogger("app.http")

app = FastAPI(title="Помічник оператора — ПР11")

INDEX_PAGE = Path(__file__).parent / "templates" / "index.html"
_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{4,64}$")


class DraftIn(BaseModel):
    request_id: str
    client: str = "anonymous"
    fresh: bool = False


@app.exception_handler(limits.RateLimited)
async def rate_limited_handler(request: Request, exc: limits.RateLimited):
    req_id = telemetry.request_id_var.get()
    return JSONResponse(
        status_code=429,
        content={"error": "rate_limited", "message": str(exc), "retry_after": exc.retry_after, "request_id": req_id},
        headers={"Retry-After": str(int(exc.retry_after))},
    )


@app.exception_handler(budget.BudgetExceeded)
async def budget_exceeded_handler(request: Request, exc: budget.BudgetExceeded):
    req_id = telemetry.request_id_var.get()
    return JSONResponse(
        status_code=429,
        content={"error": "budget", "message": str(exc), "resets_at": exc.resets_at, "request_id": req_id},
    )


@app.exception_handler(KeyError)
async def not_found_handler(request: Request, exc: KeyError):
    req_id = telemetry.request_id_var.get()
    return JSONResponse(
        status_code=404,
        content={"error": "not_found", "message": str(exc), "request_id": req_id},
    )


@app.middleware("http")
async def request_context(request: Request, call_next):
    incoming = request.headers.get("x-request-id", "")
    request_id = incoming if _REQUEST_ID.match(incoming) else telemetry.new_request_id()
    token = telemetry.request_id_var.set(request_id)
    started = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    except Exception:
        log.exception("http.unhandled", extra={"method": request.method, "path": request.url.path})
        raise
    finally:
        log.info("http.request", extra={
            "method": request.method,
            "path": request.url.path,
            "status": status,
            "duration_ms": round((time.perf_counter() - started) * 1000, 1),
        })
        telemetry.request_id_var.reset(token)


def draft_to_dict(draft: jobs.Draft) -> dict:
    data = asdict(draft)
    data["events"] = len(draft.events)
    return data


def _or_pending(fn) -> dict:
    try:
        return fn()
    except NotImplementedError as exc:
        return {"pending": str(exc)}


@app.get("/", response_class=HTMLResponse)
def page() -> str:
    return INDEX_PAGE.read_text(encoding="utf-8")


@app.get("/api/health")
def api_health() -> dict:
    return {
        "gateway": gateway.settings(),
        "limits": limits.settings(),
        "cache": cache.settings(),
        "budget": budget.settings(),
        "prompt_version": getattr(drafting, "PROMPT_VERSION", None),
        "log_file": telemetry.settings()["log_file"],
    }


@app.get("/api/usage")
def api_usage() -> dict:
    return {
        "budget": _or_pending(budget.status),
        "cache": _or_pending(cache.stats),
        "pricing": costs.settings(),
    }


@app.get("/api/requests")
def api_requests() -> list[dict]:
    return [samples.public(s) for s in samples.all_samples()]


@app.post("/api/drafts", status_code=202)
def api_create(payload: DraftIn) -> dict:
    sample = samples.get(payload.request_id)
    if not sample:
        raise KeyError(f"Звернення {payload.request_id} не знайдено")
    limits.acquire(payload.client)
    return draft_to_dict(jobs.start(sample, payload.client, fresh=payload.fresh))


@app.get("/api/drafts")
def api_drafts() -> list[dict]:
    return [draft_to_dict(d) for d in jobs.list_drafts()]


@app.get("/api/drafts/{draft_id}")
def api_draft(draft_id: str) -> dict:
    draft = jobs.get(draft_id)
    if not draft:
        raise KeyError(f"Чернетку {draft_id} не знайдено")
    return draft_to_dict(draft)


@app.get("/api/drafts/{draft_id}/events")
def api_events(draft_id: str, request: Request) -> StreamingResponse:
    try:
        since = int(request.headers.get("last-event-id") or 0)
    except ValueError:
        since = 0

    def stream() -> Iterator[str]:
        for event in jobs.subscribe(draft_id, since):
            if event is None:
                yield ": ping\n\n"
                continue
            data = json.dumps(event["data"], ensure_ascii=False)
            yield f"id: {event['id']}\nevent: {event['event']}\ndata: {data}\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/drafts/{draft_id}/cancel")
def api_cancel(draft_id: str) -> dict:
    return draft_to_dict(jobs.cancel(draft_id))
