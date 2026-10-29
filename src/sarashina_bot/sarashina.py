from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from collections import deque
from collections.abc import Callable

import httpx

log = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.sarashina.ai/v1"
# httpx's default 5s timeout is shorter than a typical non-streaming generation.
DEFAULT_TIMEOUT = 120.0


class SarashinaError(Exception):
    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class RateLimitExceeded(SarashinaError):
    def __init__(self, message: str, retry_after: int, status: int | None = None) -> None:
        super().__init__(message, status)
        self.retry_after = retry_after


class _RateLimiter:
    def __init__(self, rpm: int, clock: Callable[[], float] = time.monotonic) -> None:
        self._rpm = rpm
        self._clock = clock
        self._times: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self, require: int = 1) -> None:
        async with self._lock:
            now = self._clock()
            while self._times and now - self._times[0] >= 60:
                self._times.popleft()
            if len(self._times) + require > self._rpm:
                # The k-th oldest request has to leave the window before require slots are free.
                k = len(self._times) + require - self._rpm
                retry_after = max(1, math.ceil(self._times[k - 1] + 60 - now))
                raise RateLimitExceeded("rate limit exceeded", retry_after=retry_after)
            self._times.append(now)


class SarashinaClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        rpm: int = 5,
        clock: Callable[[], float] = time.monotonic,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        # Public so /model can switch it at runtime; every later request uses the new value.
        self.model = model
        self._limiter = _RateLimiter(rpm, clock=clock)
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=DEFAULT_TIMEOUT,
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        *,
        require_slots: int = 1,
    ) -> dict:
        await self._limiter.acquire(require_slots)
        return await self._request(messages, tools)

    async def _request(self, messages: list[dict], tools: list[dict] | None) -> dict:
        body = {"model": self.model, "messages": messages}
        if tools is not None:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        log.debug("sarashina request: %s", json.dumps(body, ensure_ascii=False))
        response = await self._post(body)
        # The API limit is a fixed one-minute window, so retrying before Retry-After is pointless.
        if response.status_code == 429:
            raise RateLimitExceeded(
                "rate limited", retry_after=_retry_after(response), status=429
            )
        if response.status_code >= 400:
            raise SarashinaError(_error_message(response), response.status_code)
        data = response.json()
        choice = data["choices"][0]
        message = choice["message"]
        tool_calls = message.get("tool_calls") or []
        # Content and arguments carry the Discord conversation, so they stay out of the pod logs.
        log.info(
            "sarashina response: finish_reason=%s tool_calls=%d names=%s",
            choice.get("finish_reason"),
            len(tool_calls),
            [call["function"]["name"] for call in tool_calls],
        )
        result = {"role": "assistant", "content": message.get("content") or ""}
        if message.get("tool_calls"):
            result["tool_calls"] = message["tool_calls"]
        return result

    async def _post(self, body: dict) -> httpx.Response:
        try:
            return await self._client.post("/chat/completions", json=body)
        except httpx.HTTPError as exc:
            raise SarashinaError(str(exc)) from exc


def _retry_after(response: httpx.Response) -> int:
    try:
        return int(response.headers["Retry-After"])
    except (KeyError, ValueError):
        # A fixed one-minute window always resets within 60 seconds.
        return 60


def _error_message(response: httpx.Response) -> str:
    try:
        message = response.json().get("message")
    except (ValueError, AttributeError):
        message = None
    if message:
        return f"{response.status_code}: {message}"
    return f"{response.status_code}: {response.text}"
