import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Iterator

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from . import gateway, jobs, limits, samples

app = FastAPI(title="Помічник оператора — ПР10")

INDEX_PAGE = Path(__file__).parent / "templates" / "index.html"


class DraftIn(BaseModel):
    request_id: str
    client: str = "anonymous"


def draft_to_dict(draft: jobs.Draft) -> dict:
    data = asdict(draft)
    data["events"] = len(draft.events)
    return data


@app.get("/", response_class=HTMLResponse)
def page() -> str:
    return INDEX_PAGE.read_text(encoding="utf-8")


@app.get("/api/health")
def api_health() -> dict:
    return {"gateway": gateway.settings(), "limits": limits.settings()}


@app.get("/api/requests")
def api_requests() -> list[dict]:
    return [samples.public(s) for s in samples.all_samples()]


@app.post("/api/drafts", status_code=202)
def api_create(payload: DraftIn):
    sample = samples.get(payload.request_id)
    if sample is None:
        return JSONResponse(
            status_code=400,
            content={"error": {"kind": "bad_request", "message": "Звернення не знайдено", "retryable": False, "request_id": payload.request_id}},
        )
    try:
        limits.acquire(payload.client)
    except limits.RateLimited as err:
        retry_seconds = max(1, int(math.ceil(err.retry_after)))
        return JSONResponse(
            status_code=429,
            headers={"Retry-After": str(retry_seconds)},
            content={"error": {"kind": "rate_limit", "message": str(err), "retryable": True}},
        )
    return draft_to_dict(jobs.start(sample, payload.client))


@app.get("/api/drafts")
def api_drafts() -> list[dict]:
    return [draft_to_dict(d) for d in jobs.list_drafts()]


@app.get("/api/drafts/{draft_id}")
def api_draft(draft_id: str):
    draft = jobs.get(draft_id)
    if draft is None:
        return JSONResponse(
            status_code=404,
            content={"error": {"kind": "not_found", "message": "Чернетку не знайдено", "retryable": False}},
        )
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
def api_cancel(draft_id: str):
    try:
        return draft_to_dict(jobs.cancel(draft_id))
    except KeyError:
        return JSONResponse(
            status_code=404,
            content={"error": {"kind": "not_found", "message": "Чернетку не знайдено", "retryable": False}},
        )
    except ValueError:
        return JSONResponse(
            status_code=409,
            content={"error": {"kind": "conflict", "message": "Неможливо скасувати завершену чернетку", "retryable": False}},
        )
