import json
import logging

import httpx
import pytest

from sarashina_bot.sarashina import (
    RateLimitExceeded,
    SarashinaClient,
    SarashinaError,
)


def make_client(handler, **kwargs):
    transport = httpx.MockTransport(handler)
    return SarashinaClient("test-key", "sarashina3-mini", transport=transport, **kwargs)


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

    assert result["content"] == "hi"
    assert set(captured["body"]) == {"model", "messages"}
    assert captured["body"]["messages"] == [{"role": "user", "content": "yo"}]
    assert captured["auth"] == "Bearer test-key"


async def test_timeout_allows_slow_generation():
    client = make_client(lambda request: ok("hi"))
    assert client._client.timeout.read >= 60


async def test_429_raises_with_retry_after_without_retrying():
    calls = {"n": 0}

    async def handler(request):
        calls["n"] += 1
        return httpx.Response(429, headers={"Retry-After": "17"})

    client = make_client(handler)
    with pytest.raises(RateLimitExceeded) as excinfo:
        await client.chat([{"role": "user", "content": "yo"}])
    assert excinfo.value.status == 429
    assert excinfo.value.retry_after == 17
    assert calls["n"] == 1


async def test_429_without_retry_after_waits_a_full_window():
    client = make_client(lambda request: httpx.Response(429))
    with pytest.raises(RateLimitExceeded) as excinfo:
        await client.chat([{"role": "user", "content": "yo"}])
    assert excinfo.value.retry_after == 60


async def test_429_with_unparseable_retry_after_waits_a_full_window():
    client = make_client(
        lambda request: httpx.Response(429, headers={"Retry-After": "soon"})
    )
    with pytest.raises(RateLimitExceeded) as excinfo:
        await client.chat([{"role": "user", "content": "yo"}])
    assert excinfo.value.retry_after == 60


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
    result = await client.chat([{"role": "user", "content": "yo"}])
    assert result["content"] == "ok"


async def test_rate_limiter_reports_seconds_until_a_slot_frees():
    now = [0.0]
    client = make_client(lambda request: ok("ok"), rpm=5, clock=lambda: now[0])
    for t in (0.0, 10.0, 20.0, 30.0, 40.0):
        now[0] = t
        await client.chat([{"role": "user", "content": "yo"}])

    now[0] = 45.5
    with pytest.raises(RateLimitExceeded) as excinfo:
        await client.chat([{"role": "user", "content": "yo"}])
    # The oldest request (t=0) leaves the window at t=60.
    assert excinfo.value.retry_after == 15


async def test_rate_limiter_retry_after_waits_for_enough_slots():
    now = [0.0]
    client = make_client(lambda request: ok("ok"), rpm=5, clock=lambda: now[0])
    for t in (0.0, 10.0, 20.0, 30.0):
        now[0] = t
        await client.chat([{"role": "user", "content": "yo"}])

    now[0] = 40.0
    with pytest.raises(RateLimitExceeded) as excinfo:
        await client.chat([{"role": "user", "content": "yo"}], require_slots=3)
    # Two of the four must expire: the second one (t=10) leaves at t=70.
    assert excinfo.value.retry_after == 30


async def test_rate_limiter_retry_after_is_at_least_one_second():
    now = [0.0]
    client = make_client(lambda request: ok("ok"), rpm=1, clock=lambda: now[0])
    await client.chat([{"role": "user", "content": "yo"}])

    now[0] = 59.999
    with pytest.raises(RateLimitExceeded) as excinfo:
        await client.chat([{"role": "user", "content": "yo"}])
    assert excinfo.value.retry_after == 1


TOOLS = [{"type": "function", "function": {"name": "web_search"}}]


async def test_body_with_tools_sends_tool_choice_auto():
    captured = {}

    async def handler(request):
        captured["body"] = json.loads(request.content)
        return ok("hi")

    client = make_client(handler)
    await client.chat([{"role": "user", "content": "yo"}], tools=TOOLS)

    assert set(captured["body"]) == {"model", "messages", "tools", "tool_choice"}
    assert captured["body"]["tools"] == TOOLS
    assert captured["body"]["tool_choice"] == "auto"


async def test_tool_calls_are_returned():
    tool_calls = [
        {
            "id": "call-1",
            "type": "function",
            "function": {"name": "web_search", "arguments": '{"query": "q"}'},
        }
    ]

    async def handler(request):
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"role": "assistant", "content": "", "tool_calls": tool_calls}}
                ]
            },
        )

    client = make_client(handler)
    result = await client.chat([{"role": "user", "content": "yo"}], tools=TOOLS)

    assert result == {"role": "assistant", "content": "", "tool_calls": tool_calls}


async def test_no_tool_calls_key_when_absent():
    client = make_client(lambda request: ok("hi"))
    result = await client.chat([{"role": "user", "content": "yo"}])
    assert result == {"role": "assistant", "content": "hi"}


async def test_null_content_becomes_empty_string():
    client = make_client(lambda request: ok(None))
    result = await client.chat([{"role": "user", "content": "yo"}])
    assert result["content"] == ""


async def test_require_slots_checks_free_slots_up_front():
    client = make_client(lambda request: ok("ok"), rpm=5, clock=lambda: 0.0)
    for _ in range(4):
        await client.chat([{"role": "user", "content": "yo"}])

    with pytest.raises(RateLimitExceeded):
        await client.chat([{"role": "user", "content": "yo"}], require_slots=2)
    result = await client.chat([{"role": "user", "content": "yo"}], require_slots=1)
    assert result["content"] == "ok"


async def test_response_metadata_is_logged_without_content(caplog):
    tool_calls = [
        {
            "id": "call-1",
            "type": "function",
            "function": {"name": "run_command", "arguments": '{"command": "secret-arg"}'},
        }
    ]

    async def handler(request):
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": "secret-content",
                            "tool_calls": tool_calls,
                        },
                    }
                ]
            },
        )

    client = make_client(handler)
    with caplog.at_level(logging.INFO, logger="sarashina_bot.sarashina"):
        await client.chat([{"role": "user", "content": "yo"}], tools=TOOLS)

    logged = "\n".join(r.getMessage() for r in caplog.records if r.levelno == logging.INFO)
    assert "finish_reason=tool_calls" in logged
    assert "run_command" in logged
    assert "secret-content" not in logged
    assert "secret-arg" not in logged


async def test_response_without_tool_calls_is_logged(caplog):
    async def handler(request):
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"finish_reason": "stop", "message": {"content": "secret-content"}}
                ]
            },
        )

    client = make_client(handler)
    with caplog.at_level(logging.INFO, logger="sarashina_bot.sarashina"):
        await client.chat([{"role": "user", "content": "yo"}])

    logged = "\n".join(r.getMessage() for r in caplog.records if r.levelno == logging.INFO)
    assert "finish_reason=stop" in logged
    assert "secret-content" not in logged
