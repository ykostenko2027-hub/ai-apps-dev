import datetime
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Iterator

from . import drafting, gateway
from .providers.base import ProviderError

TERMINAL = {"done", "failed", "cancelled"}


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
    events: list[dict] = field(default_factory=list)


_drafts: dict[str, Draft] = {}
_order: list[str] = []
_cancels: dict[str, threading.Event] = {}
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
        if len(_order) > 200:
            removed_id = _order.pop()
            _drafts.pop(removed_id, None)
    return draft


def get(draft_id: str) -> Draft | None:
    with _lock:
        return _drafts.get(draft_id)


def list_drafts(limit: int = 50) -> list[Draft]:
    with _lock:
        return [_drafts[i] for i in _order[:limit]]


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


def _worker(draft: Draft, sample: dict) -> None:
    t0 = time.perf_counter()
    cancel_event = _cancels.get(draft.draft_id)
    is_cancelled = lambda: cancel_event.is_set() if cancel_event else False

    def on_delta(delta: str) -> None:
        with _lock:
            draft.text += delta
            if "first_token" not in draft.elapsed:
                draft.elapsed["first_token"] = round(time.perf_counter() - t0, 2)
        emit(draft, "delta", {"text": delta})

    def on_status(data: dict) -> None:
        with _lock:
            if "status" in data:
                draft.status = data["status"]
            if "attempt" in data:
                draft.attempts = data["attempt"]
            if "provider" in data:
                draft.provider = data["provider"]
            if "fallback" in data:
                draft.fallback = data["fallback"]
            draft.updated_at = _now()
        emit(draft, "status", data)

    try:
        req = drafting.build_request(sample)
        res = gateway.generate(req, on_delta, on_status, is_cancelled)
        if is_cancelled():
            with _lock:
                draft.status = "cancelled"
                draft.updated_at = _now()
            emit(draft, "cancelled", {})
            return

        warnings = drafting.check(draft.text, sample)
        with _lock:
            draft.status = "done"
            draft.provider = res.provider
            draft.model = res.model
            draft.usage = res.usage
            draft.warnings = warnings
            draft.elapsed["total"] = round(time.perf_counter() - t0, 2)
            draft.updated_at = _now()
        emit(draft, "done", {
            "text": draft.text,
            "provider": res.provider,
            "model": res.model,
            "usage": res.usage,
            "elapsed": draft.elapsed,
            "attempts": draft.attempts,
            "fallback": draft.fallback,
            "warnings": draft.warnings,
        })
    except Exception as err:
        if is_cancelled() or (isinstance(err, ProviderError) and err.kind == "cancelled"):
            with _lock:
                draft.status = "cancelled"
                draft.updated_at = _now()
            emit(draft, "cancelled", {})
        else:
            with _lock:
                draft.status = "failed"
                if isinstance(err, ProviderError):
                    draft.error = {"kind": err.kind, "message": err.message, "retryable": err.retryable}
                else:
                    draft.error = {"kind": "unknown", "message": str(err), "retryable": False}
                draft.updated_at = _now()
            emit(draft, "error", draft.error)
    finally:
        _cancels.pop(draft.draft_id, None)


def start(sample: dict, client: str) -> Draft:
    draft = new_draft(sample["id"], client)
    event = threading.Event()
    with _lock:
        _cancels[draft.draft_id] = event
    thread = threading.Thread(target=_worker, args=(draft, sample), daemon=True)
    thread.start()
    return draft


def cancel(draft_id: str) -> Draft:
    draft = get(draft_id)
    if draft is None:
        raise KeyError("Чернетку не знайдено")
    if draft.status in TERMINAL:
        raise ValueError("Чернетка вже завершена")
    event = _cancels.get(draft_id)
    if event:
        event.set()
    with _lock:
        draft.status = "cancelled"
        draft.updated_at = _now()
    emit(draft, "cancelled", {})
    return draft
