from __future__ import annotations

import logging

import httpx

log = logging.getLogger(__name__)

EXA_SEARCH_URL = "https://api.exa.ai/search"
DEFAULT_TIMEOUT = 30.0


class ExaError(Exception):
    pass


class ExaClient:
    def __init__(
        self,
        api_key: str,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._client = httpx.AsyncClient(
            headers={"x-api-key": api_key},
            timeout=timeout,
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def search(self, query: str, num_results: int = 5) -> str:
        body = {
            "query": query,
            "numResults": num_results,
            "contents": {"highlights": {"maxCharacters": 500}},
        }
        log.debug("exa request: %s", body)
        try:
            response = await self._client.post(EXA_SEARCH_URL, json=body)
        except httpx.HTTPError as exc:
            raise ExaError(str(exc)) from exc
        if response.status_code >= 400:
            raise ExaError(f"{response.status_code}: {response.text}")
        try:
            return _format_results(response.json()["results"])
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            raise ExaError(f"unexpected response: {exc!r}") from exc


def _format_results(results: list[dict]) -> str:
    blocks: list[str] = []
    for index, result in enumerate(results, 1):
        highlights = result.get("highlights") or []
        blocks.append(
            "\n".join(
                [f"[{index}] {result.get('title', '')}", result.get("url", "")]
                + list(highlights)
            )
        )
    return "\n\n".join(blocks)
