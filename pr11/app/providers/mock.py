import os
import random
import time
from typing import Iterator
from dotenv import load_dotenv

from .base import Chunk, ProviderError, Request, Result
from .. import samples

load_dotenv()

FIRST_DELAY_MS = int(os.getenv("MOCK_FIRST_DELAY_MS", "600"))
DELAY_MS = int(os.getenv("MOCK_DELAY_MS", "40"))
FAILURE_RATE = float(os.getenv("MOCK_FAILURE_RATE", "0"))
FAILURE_KIND = os.getenv("MOCK_FAILURE_KIND", "unavailable")


class MockProvider:
    name = "mock"
    model = "mock-1"

    def _get_text(self, request: Request) -> str:
        req_id = request.metadata.get("request_id")
        if req_id:
            s = samples.get(req_id)
            if s and s.get("reply_example"):
                return s["reply_example"]
        return "Добрий день! Дякуємо за звернення до служби підтримки «Сузірʼя». Ваше питання опрацьовується."

    def _calc_usage(self, request: Request, text: str) -> dict:
        prompt_tokens = max(10, sum(len(m.get("content", "")) // 3 for m in request.messages))
        completion_tokens = max(10, len(text) // 3)
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

    def _check_failure(self) -> None:
        failure_rate = float(os.getenv("MOCK_FAILURE_RATE", str(FAILURE_RATE)))
        failure_kind = os.getenv("MOCK_FAILURE_KIND", FAILURE_KIND)
        if failure_rate > 0 and random.random() < failure_rate:
            if failure_kind == "timeout":
                raise ProviderError(kind="timeout", message="Час очікування відповіді провайдера вичерпано", retryable=True)
            if failure_kind == "rate_limit":
                raise ProviderError(kind="rate_limit", message="Перевищено ліміт запитів до провайдера", retryable=True, retry_after=2.0)
            if failure_kind == "auth":
                raise ProviderError(kind="auth", message="Помилка автентифікації провайдера", retryable=False)
            if failure_kind == "bad_request":
                raise ProviderError(kind="bad_request", message="Некоректний запит до моделі", retryable=False)
            raise ProviderError(kind="unavailable", message="Провайдер тимчасово недоступний", retryable=True)

    def complete(self, request: Request) -> Result:
        first_delay = int(os.getenv("MOCK_FIRST_DELAY_MS", str(FIRST_DELAY_MS))) / 1000.0
        time.sleep(first_delay)
        self._check_failure()

        text = self._get_text(request)
        usage = self._calc_usage(request, text)
        return Result(
            text=text,
            provider=self.name,
            model=self.model,
            usage=usage,
            finish_reason="stop",
            elapsed=first_delay,
            cached=False,
        )

    def stream(self, request: Request) -> Iterator[Chunk]:
        first_delay = int(os.getenv("MOCK_FIRST_DELAY_MS", str(FIRST_DELAY_MS))) / 1000.0
        delay = int(os.getenv("MOCK_DELAY_MS", str(DELAY_MS))) / 1000.0
        time.sleep(first_delay)
        self._check_failure()

        text = self._get_text(request)
        usage = self._calc_usage(request, text)
        words = text.split(" ")
        failure_rate = float(os.getenv("MOCK_FAILURE_RATE", str(FAILURE_RATE)))
        failure_kind = os.getenv("MOCK_FAILURE_KIND", FAILURE_KIND)
        fail_mid = (failure_rate > 0 and failure_kind == "mid_stream" and random.random() < failure_rate)

        for i, word in enumerate(words):
            chunk_text = word if i == len(words) - 1 else word + " "
            if i > 2 and fail_mid:
                raise ProviderError(kind="unavailable", message="Зʼєднання обірвано посеред потоку", retryable=True)
            time.sleep(delay)
            if i == len(words) - 1:
                yield Chunk(text=chunk_text, usage=usage, finish_reason="stop")
            else:
                yield Chunk(text=chunk_text)
