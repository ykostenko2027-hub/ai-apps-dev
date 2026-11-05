import hashlib
import json
import os
import threading
import time
from pathlib import Path
from dotenv import load_dotenv

from .providers.base import Request, Result

load_dotenv()

ENABLED = os.getenv("CACHE_ENABLED", "1") == "1"
TTL_SECONDS = float(os.getenv("CACHE_TTL_SECONDS", "3600"))
MAX_ENTRIES = int(os.getenv("CACHE_MAX_ENTRIES", "500"))
CACHE_FILE = os.getenv("CACHE_FILE", "state/cache.json")

_lock = threading.Lock()
_hits = 0
_misses = 0
_bypassed = 0
_store: dict[str, dict] = {}


def _load_disk() -> None:
    global _store
    if not CACHE_FILE:
        return
    p = Path(CACHE_FILE)
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            now = time.time()
            _store = {k: v for k, v in data.items() if v.get("expires_at", 0) > now}
        except Exception:
            _store = {}


def _save_disk() -> None:
    if not CACHE_FILE:
        return
    p = Path(CACHE_FILE)
    p.parent.mkdir(parents=True, exist_ok=True)
    temp_path = p.with_suffix(".tmp")
    try:
        temp_path.write_text(json.dumps(_store, ensure_ascii=False), encoding="utf-8")
        temp_path.replace(p)
    except Exception:
        pass


_load_disk()


def settings() -> dict:
    return {
        "enabled": ENABLED,
        "ttl_seconds": TTL_SECONDS,
        "max_entries": MAX_ENTRIES,
        "file": CACHE_FILE or None,
    }


def record_bypass() -> None:
    global _bypassed
    with _lock:
        _bypassed += 1


def make_key(request: Request, prompt_version: str) -> str:
    key_dict = {
        "model": request.model or "",
        "prompt_version": prompt_version,
        "messages": request.messages,
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
    }
    raw = json.dumps(key_dict, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def get(key: str) -> Result | None:
    global _hits, _misses
    if not ENABLED:
        return None

    with _lock:
        item = _store.get(key)
        now = time.time()
        if item is None:
            _misses += 1
            return None
        if item.get("expires_at", 0) <= now:
            del _store[key]
            _save_disk()
            _misses += 1
            return None

        _hits += 1
        res_dict = item.get("result", {})
        return Result(
            text=res_dict.get("text", ""),
            provider=res_dict.get("provider", "cache"),
            model=res_dict.get("model", ""),
            usage={"prompt_tokens": 0, "completion_tokens": 0},
            finish_reason=res_dict.get("finish_reason", "stop"),
            elapsed=0.0,
            cost=0.0,
            cost_estimated=False,
            cached=True,
        )


def put(key: str, result: Result) -> None:
    if not ENABLED or result is None or result.cached:
        return

    with _lock:
        if len(_store) >= MAX_ENTRIES and key not in _store:
            oldest_key = min(_store.keys(), key=lambda k: _store[k].get("expires_at", 0))
            del _store[oldest_key]

        _store[key] = {
            "expires_at": time.time() + TTL_SECONDS,
            "result": {
                "text": result.text,
                "provider": result.provider,
                "model": result.model,
                "finish_reason": result.finish_reason,
            }
        }
        _save_disk()


def stats() -> dict:
    with _lock:
        now = time.time()
        active = sum(1 for v in _store.values() if v.get("expires_at", 0) > now)
        return {
            "hits": _hits,
            "misses": _misses,
            "bypassed": _bypassed,
            "entries": active,
        }
