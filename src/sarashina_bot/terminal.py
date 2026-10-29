from __future__ import annotations

import logging

import httpx

log = logging.getLogger(__name__)

# Longer than the sandbox's own 30s command timeout, so the sandbox reports the timeout itself.
DEFAULT_TIMEOUT = 60.0


class TerminalError(Exception):
    pass


class TerminalClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"), timeout=timeout, transport=transport
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def run(self, command: str) -> str:
        log.info("terminal command: %s", command)
        try:
            response = await self._client.post("/run", json={"command": command})
        except httpx.HTTPError as exc:
            raise TerminalError(str(exc)) from exc
        if response.status_code >= 400:
            raise TerminalError(f"{response.status_code}: {response.text}")
        try:
            data = response.json()
            return _format_result(data["exit_code"], data["timed_out"], data["output"])
        except (ValueError, KeyError, TypeError) as exc:
            raise TerminalError(f"unexpected response: {exc!r}") from exc


def _format_result(exit_code: int, timed_out: bool, output: str) -> str:
    status = "タイムアウトで停止しました" if timed_out else f"終了コード {exit_code}"
    return f"{status}\n{output}"
