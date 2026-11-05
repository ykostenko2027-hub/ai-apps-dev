import time
from typing import Iterator

import openai
from openai import OpenAI

from .base import Chunk, ProviderError, Request, Result


class OpenAICompatProvider:
    name = "openai"

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key or "missing-key",
            timeout=self.timeout,
            max_retries=0,
        )

    def _convert_error(self, err: Exception) -> ProviderError:
        if isinstance(err, openai.APITimeoutError):
            return ProviderError("timeout", "Таймаут очікування відповіді від моделі", retryable=True, detail=str(err))
        if isinstance(err, openai.RateLimitError):
            retry_after = None
            if hasattr(err, "response") and err.response is not None:
                h = err.response.headers.get("retry-after")
                if h:
                    try:
                        retry_after = float(h)
                    except ValueError:
                        pass
            return ProviderError("rate_limit", "Перевищено ліміт запитів провайдера", retryable=True, retry_after=retry_after, detail=str(err))
        if isinstance(err, openai.APIConnectionError):
            return ProviderError("unavailable", "Помилка з'єднання з сервісом моделі", retryable=True, detail=str(err))
        if isinstance(err, openai.AuthenticationError):
            return ProviderError("auth", "Помилка автентифікації API-ключа", retryable=False, detail=str(err))
        if isinstance(err, openai.BadRequestError):
            return ProviderError("bad_request", "Некоректний запит до моделі", retryable=False, detail=str(err))
        if isinstance(err, openai.InternalServerError):
            return ProviderError("unavailable", "Сервіс моделі тимчасово недоступний", retryable=True, detail=str(err))
        if isinstance(err, openai.APIStatusError):
            if err.status_code >= 500:
                return ProviderError("unavailable", "Серверна помилка провайдера", retryable=True, detail=str(err))
            if err.status_code == 429:
                return ProviderError("rate_limit", "Перевищено ліміт запитів", retryable=True, detail=str(err))
            if err.status_code in (401, 403):
                return ProviderError("auth", "Помилка авторизації", retryable=False, detail=str(err))
            return ProviderError("bad_request", "Помилка запиту", retryable=False, detail=str(err))
        return ProviderError("unknown", f"Непередбачений збій провайдера: {err}", retryable=False, detail=str(err))

    def complete(self, request: Request) -> Result:
        t0 = time.perf_counter()
        target_model = request.model or self.model
        try:
            response = self.client.chat.completions.create(
                model=target_model,
                messages=request.messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=False,
            )
        except Exception as e:
            raise self._convert_error(e) from None

        choice = response.choices[0]
        text = choice.message.content or ""
        finish_reason = choice.finish_reason
        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        elapsed = round(time.perf_counter() - t0, 2)
        return Result(
            text=text,
            provider=self.name,
            model=target_model,
            usage=usage,
            finish_reason=finish_reason,
            elapsed=elapsed,
        )

    def stream(self, request: Request) -> Iterator[Chunk]:
        target_model = request.model or self.model
        try:
            stream_resp = self.client.chat.completions.create(
                model=target_model,
                messages=request.messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=True,
                stream_options={"include_usage": True},
            )
        except Exception as e:
            raise self._convert_error(e) from None

        try:
            for part in stream_resp:
                text = ""
                finish_reason = None
                if part.choices:
                    delta = part.choices[0].delta
                    if delta and delta.content:
                        text = delta.content
                    finish_reason = part.choices[0].finish_reason
                usage = None
                if hasattr(part, "usage") and part.usage:
                    usage = {
                        "prompt_tokens": part.usage.prompt_tokens,
                        "completion_tokens": part.usage.completion_tokens,
                        "total_tokens": part.usage.total_tokens,
                    }
                yield Chunk(text=text, usage=usage, finish_reason=finish_reason)
        except Exception as e:
            raise self._convert_error(e) from None
        finally:
            try:
                stream_resp.close()
            except Exception:
                pass
