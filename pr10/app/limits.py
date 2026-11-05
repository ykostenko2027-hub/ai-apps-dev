import os
import threading
import time
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "10"))

_history: dict[str, list[float]] = defaultdict(list)
_lock = threading.Lock()


class RateLimited(Exception):
    def __init__(self, retry_after: float) -> None:
        super().__init__(f"ліміт вичерпано, повторіть через {retry_after:.0f} с")
        self.retry_after = retry_after


def settings() -> dict:
    return {"per_minute": PER_MINUTE}


def acquire(client: str) -> None:
    now = time.monotonic()
    window = 60.0
    with _lock:
        timestamps = _history[client]
        cutoff = now - window
        _history[client] = [t for t in timestamps if t > cutoff]
        if len(_history[client]) >= PER_MINUTE:
            oldest = _history[client][0]
            retry_after = max(1.0, window - (now - oldest))
            raise RateLimited(retry_after)
        _history[client].append(now)
