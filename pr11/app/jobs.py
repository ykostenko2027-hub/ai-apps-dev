import contextvars
import datetime
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Iterator

from . import drafting, gateway, telemetry
from .providers.base import ProviderError

TERMINAL = {"done", "failed", "cancelled"}

log = logging.getLogger("app.jobs")


@dataclass
class Draft:
    draft_id: str
    request_id: str
    client: str
    status: str = "queued"
    text: str = ""
    attempts: int = 0
    provider: str | None = None
    model: str | None = None
    fallback: bool = False
    error: dict | None = None
    usage: dict | None = None
    elapsed: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    cost: float | None = None
    cost_estimated: bool = False
    cached: bool = False
    events: list[dict] = field(default_factory=list)


_drafts: dict[str, Draft] = {}
_order: list[str] = []
_cancelled: dict[str, bool] = {}
_lock = threading.Lock()
_changed = threading.Condition(_lock)


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def new_draft(request_id: str, client: str) -> Draft:
    draft = Draft(
        draft_id=uuid.uuid4().hex[:8],
        request_id=request_id,
        client=client,
        created_at=_now(),
        updated_at=_now(),
    )
    with _lock:
        _drafts[draft.draft_id] = draft
        _order.insert(0, draft.draft_id)
        _cancelled[draft.draft_id] = False
    return draft


def get(draft_id: str) -> Draft | None:
    with _lock:
        return _drafts.get(draft_id)


def list_drafts(limit: int = 50) -> list[Draft]:
    with _lock:
        return [_drafts[i] for i in _order[:limit] if i in _drafts]


def emit(draft: Draft, event: str, data: dict | None = None) -> None:
    with _changed:
        draft.events.append({"id": len(draft.events) + 1, "event": event, "data": data or {}})
        draft.updated_at = _now()
        _changed.notify_all()


def subscribe(draft_id: str, since: int = 0, heartbeat: float = 15.0) -> Iterator[dict | None]:
    while True:
        with _changed:
            draft = _drafts.get(draft_id)
            if draft is None:
                return
            pending = draft.events[since:]
            if not pending:
                if draft.status in TERMINAL:
                    return
                _changed.wait(timeout=heartbeat)
                pending = draft.events[since:]
        if not pending:
            yield None
            continue
        for event in pending:
            yield event
        since += len(pending)


def _worker(draft: Draft, sample: dict, fresh: bool, req_id: str) -> None:
    telemetry.request_id_var.set(req_id)
    started = time.perf_counter()
    first_token_time = None

    with _lock:
        if _cancelled.get(draft.draft_id):
            draft.status = "cancelled"
            emit(draft, "cancelled", {})
            return
        draft.status = "running"
    emit(draft, "status", {"status": "running", "attempt": 1})

    req = drafting.build_request(sample)
    req.metadata.update({
        "draft_id": draft.draft_id,
        "client": draft.client,
        "fresh": fresh,
        "http_request_id": req_id,
    })

    def on_chunk(chunk_text: str) -> None:
        nonlocal first_token_time
        if first_token_time is None:
            first_token_time = time.perf_counter() - started
        with _lock:
            draft.text += chunk_text
        emit(draft, "delta", {"text": chunk_text})

    def is_cancelled() -> bool:
        with _lock:
            return _cancelled.get(draft.draft_id, False)

    try:
        res = gateway.generate_stream(req, on_chunk, is_cancelled)
        total_time = time.perf_counter() - started
        warnings = drafting.check(draft.text, sample)

        with _lock:
            draft.status = "done"
            draft.provider = res.provider
            draft.model = res.model
            draft.usage = res.usage
            draft.cost = res.cost
            draft.cost_estimated = res.cost_estimated
            draft.cached = res.cached
            draft.attempts = 1
            draft.warnings = warnings
            draft.elapsed = {
                "first_token": round(first_token_time or total_time, 2),
                "total": round(total_time, 2),
            }

        done_data = {
            "text": draft.text,
            "provider": draft.provider,
            "model": draft.model,
            "usage": draft.usage,
            "elapsed": draft.elapsed,
            "attempts": draft.attempts,
            "fallback": draft.fallback,
            "warnings": draft.warnings,
            "cached": draft.cached,
            "cost": draft.cost,
            "cost_estimated": draft.cost_estimated,
        }
        emit(draft, "done", done_data)

        log.info("draft.done", extra={
            "draft_id": draft.draft_id,
            "request_id": req_id,
            "client": draft.client,
            "attempts": draft.attempts,
            "provider": draft.provider,
            "model": draft.model,
            "cached": draft.cached,
            "cost": draft.cost,
            "cost_estimated": draft.cost_estimated,
            "duration_ms": round(total_time * 1000, 1),
        })

    except ProviderError as pe:
        total_time = time.perf_counter() - started
        if pe.kind == "cancelled" or is_cancelled():
            with _lock:
                draft.status = "cancelled"
            emit(draft, "cancelled", {})
            log.info("draft.cancelled", extra={
                "draft_id": draft.draft_id,
                "request_id": req_id,
                "client": draft.client,
                "duration_ms": round(total_time * 1000, 1),
            })
        else:
            with _lock:
                draft.status = "failed"
                draft.error = {"kind": pe.kind, "message": pe.message, "retryable": pe.retryable}
            emit(draft, "error", {"kind": pe.kind, "message": pe.message, "retryable": pe.retryable})
            log.error("draft.failed", extra={
                "draft_id": draft.draft_id,
                "request_id": req_id,
                "client": draft.client,
                "kind": pe.kind,
                "message": pe.message,
                "duration_ms": round(total_time * 1000, 1),
            })

    except Exception as exc:
        total_time = time.perf_counter() - started
        with _lock:
            draft.status = "failed"
            draft.error = {"kind": "unknown", "message": str(exc), "retryable": False}
        emit(draft, "error", {"kind": "unknown", "message": str(exc), "retryable": False})
        log.exception("draft.failed", extra={
            "draft_id": draft.draft_id,
            "request_id": req_id,
            "client": draft.client,
            "kind": "unknown",
            "duration_ms": round(total_time * 1000, 1),
        })


def start(sample: dict, client: str, fresh: bool = False) -> Draft:
    draft = new_draft(sample["id"], client)
    req_id = telemetry.request_id_var.get() or draft.draft_id

    log.info("draft.created", extra={
        "draft_id": draft.draft_id,
        "sample_id": draft.request_id,
        "request_id": req_id,
        "client": client,
        "fresh": fresh,
    })

    ctx = contextvars.copy_context()
    thread = threading.Thread(target=ctx.run, args=(_worker, draft, sample, fresh, req_id), daemon=True)
    thread.start()
    return draft


def cancel(draft_id: str) -> Draft:
    with _lock:
        draft = _drafts.get(draft_id)
        if draft is None:
            raise KeyError("Чернетку не знайдено")
        _cancelled[draft_id] = True
        if draft.status not in TERMINAL:
            draft.status = "cancelled"
            emit(draft, "cancelled", {})
    return draft
