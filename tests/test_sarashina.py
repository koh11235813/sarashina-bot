import json

import httpx
import pytest

from sarashina_bot.sarashina import (
    RateLimitExceeded,
    SarashinaClient,
    SarashinaError,
)


def make_client(handler, **kwargs):
    transport = httpx.MockTransport(handler)
    return SarashinaClient("test-key", "sarashina3-mini", transport=transport, backoff=0, **kwargs)


def ok(content):
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


async def test_body_whitelist_and_auth_header():
    captured = {}

    async def handler(request):
        captured["body"] = json.loads(request.content)
        captured["auth"] = request.headers.get("authorization")
        return ok("hi")

    client = make_client(handler)
    result = await client.chat([{"role": "user", "content": "yo"}])

    assert result == "hi"
    assert set(captured["body"]) == {"model", "messages"}
    assert captured["body"]["messages"] == [{"role": "user", "content": "yo"}]
    assert captured["auth"] == "Bearer test-key"


async def test_timeout_allows_slow_generation():
    client = make_client(lambda request: ok("hi"))
    assert client._client.timeout.read >= 60


async def test_429_then_retry_succeeds():
    calls = {"n": 0}

    async def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429)
        return ok("recovered")

    client = make_client(handler)
    assert await client.chat([{"role": "user", "content": "yo"}]) == "recovered"
    assert calls["n"] == 2


async def test_429_twice_raises():
    calls = {"n": 0}

    async def handler(request):
        calls["n"] += 1
        return httpx.Response(429)

    client = make_client(handler)
    with pytest.raises(SarashinaError):
        await client.chat([{"role": "user", "content": "yo"}])
    assert calls["n"] == 2


async def test_400_message_is_included():
    async def handler(request):
        return httpx.Response(400, json={"message": "x"})

    client = make_client(handler)
    with pytest.raises(SarashinaError) as excinfo:
        await client.chat([{"role": "user", "content": "yo"}])
    assert "x" in str(excinfo.value)
    assert excinfo.value.status == 400


async def test_400_non_json_body_tolerated():
    async def handler(request):
        return httpx.Response(400, text="boom")

    client = make_client(handler)
    with pytest.raises(SarashinaError) as excinfo:
        await client.chat([{"role": "user", "content": "yo"}])
    assert "boom" in str(excinfo.value)


async def test_rate_limiter_sliding_window():
    now = [0.0]

    async def handler(request):
        return ok("ok")

    client = make_client(handler, rpm=5, clock=lambda: now[0])
    for _ in range(5):
        await client.chat([{"role": "user", "content": "yo"}])

    with pytest.raises(RateLimitExceeded):
        await client.chat([{"role": "user", "content": "yo"}])

    now[0] = 61.0
    assert await client.chat([{"role": "user", "content": "yo"}]) == "ok"
