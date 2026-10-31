import json
import logging

import httpx
import pytest

from sarashina_bot.sarashina import (
    REASONING_EFFORT_MODEL,
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

    assert result == {
        "role": "assistant",
        "content": "",
        "tool_calls": tool_calls,
        "finish_reason": None,
    }


async def test_no_tool_calls_key_when_absent():
    client = make_client(lambda request: ok("hi"))
    result = await client.chat([{"role": "user", "content": "yo"}])
    assert result == {"role": "assistant", "content": "hi", "finish_reason": None}


async def test_finish_reason_is_returned():
    async def handler(request):
        return httpx.Response(
            200,
            json={"choices": [{"finish_reason": "length", "message": {"content": ""}}]},
        )

    client = make_client(handler)
    result = await client.chat([{"role": "user", "content": "yo"}])
    assert result["finish_reason"] == "length"


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


async def sent_body(model, reasoning_effort):
    captured = {}

    async def handler(request):
        captured["body"] = json.loads(request.content)
        return ok("hi")

    client = make_client(handler)
    client.model = model
    client.reasoning_effort = reasoning_effort
    await client.chat([{"role": "user", "content": "yo"}])
    return captured["body"]


async def test_reasoning_effort_is_sent_to_thinking_model():
    body = await sent_body(REASONING_EFFORT_MODEL, "high")
    assert body["reasoning_effort"] == "high"


async def test_reasoning_effort_is_not_sent_to_other_models():
    body = await sent_body("sarashina3-mini", "high")
    assert "reasoning_effort" not in body


async def test_reasoning_effort_is_not_sent_when_unset():
    body = await sent_body(REASONING_EFFORT_MODEL, None)
    assert "reasoning_effort" not in body


async def test_reasoning_effort_needs_exact_model_name():
    body = await sent_body("foo-thinking", "high")
    assert "reasoning_effort" not in body


async def test_reasoning_effort_is_unset_at_start():
    client = make_client(lambda request: ok("hi"))
    assert client.reasoning_effort is None


def translated(text):
    return httpx.Response(200, json={"translated_text": text, "extra": 1})


async def test_translate_request_path_body_and_auth_header():
    captured = {}

    async def handler(request):
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        captured["auth"] = request.headers.get("authorization")
        return translated("Hello")

    client = make_client(handler)
    result = await client.translate("こんにちは", "ja", "en")

    assert result == "Hello"
    assert captured["path"] == "/v1/translate"
    assert captured["body"] == {
        "text": "こんにちは",
        "source_lang": "ja",
        "target_lang": "en",
    }
    assert captured["auth"] == "Bearer test-key"


async def test_translate_429_raises_with_retry_after():
    client = make_client(
        lambda request: httpx.Response(429, headers={"Retry-After": "17"})
    )
    with pytest.raises(RateLimitExceeded) as excinfo:
        await client.translate("yo", "en", "ja")
    assert excinfo.value.status == 429
    assert excinfo.value.retry_after == 17


async def test_translate_429_without_retry_after_waits_a_full_window():
    client = make_client(lambda request: httpx.Response(429))
    with pytest.raises(RateLimitExceeded) as excinfo:
        await client.translate("yo", "en", "ja")
    assert excinfo.value.retry_after == 60


async def test_translate_400_raises_with_message():
    client = make_client(
        lambda request: httpx.Response(400, json={"message": "same language"})
    )
    with pytest.raises(SarashinaError) as excinfo:
        await client.translate("yo", "en", "ja")
    assert excinfo.value.status == 400
    assert "same language" in str(excinfo.value)


async def test_translate_bad_json_raises_sarashina_error():
    client = make_client(lambda request: httpx.Response(200, text="not json"))
    with pytest.raises(SarashinaError):
        await client.translate("yo", "en", "ja")


async def test_translate_missing_field_raises_sarashina_error():
    client = make_client(lambda request: httpx.Response(200, json={"foo": "bar"}))
    with pytest.raises(SarashinaError):
        await client.translate("yo", "en", "ja")


async def test_translate_network_error_raises_sarashina_error():
    def handler(request):
        raise httpx.ConnectError("down")

    client = make_client(handler)
    with pytest.raises(SarashinaError):
        await client.translate("yo", "en", "ja")


def by_path(request):
    if request.url.path.endswith("/translate"):
        return translated("t")
    return ok("c")


async def test_translate_has_its_own_rate_limiter():
    client = make_client(by_path, rpm=1, clock=lambda: 0.0)
    await client.translate("yo", "en", "ja")

    with pytest.raises(RateLimitExceeded) as excinfo:
        await client.translate("yo", "en", "ja")
    assert excinfo.value.retry_after == 60


async def test_chat_does_not_use_translate_slots():
    client = make_client(by_path, rpm=1, clock=lambda: 0.0)
    await client.chat([{"role": "user", "content": "yo"}])

    assert await client.translate("yo", "en", "ja") == "t"


async def test_translate_does_not_use_chat_slots():
    client = make_client(by_path, rpm=1, clock=lambda: 0.0)
    await client.translate("yo", "en", "ja")

    result = await client.chat([{"role": "user", "content": "yo"}])
    assert result["content"] == "c"


async def test_translate_does_not_log_text(caplog):
    client = make_client(lambda request: translated("secret-out"))
    with caplog.at_level(logging.DEBUG, logger="sarashina_bot.sarashina"):
        await client.translate("secret-in", "ja", "en")

    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "secret-in" not in logged
    assert "secret-out" not in logged


async def test_embed_body_and_result():
    captured = {}

    def handler(request):
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"data": [{"embedding": [0.1, 0.2]}]})

    client = make_client(handler)
    assert await client.embed("text", "RETRIEVAL_QUERY") == [0.1, 0.2]
    assert captured == {
        "path": "/v1/embeddings",
        "body": {"input": "text", "task_type": "RETRIEVAL_QUERY", "model": "sarashina3-embedding"},
    }


async def test_rerank_body_and_ids_in_response_order():
    captured = {}

    def handler(request):
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"data": [{"result": [
            {"document_id": 2, "score": 0.1}, {"document_id": 0, "score": 0.9},
        ]}]})

    client = make_client(handler)
    assert await client.rerank("q", ["a", "b", "c"]) == [2, 0]
    assert captured == {
        "path": "/v1/rerank",
        "body": {"query": "q", "documents": ["a", "b", "c"], "task_type": "GENERAL_QA", "model": "sarashina3-rerank"},
    }


@pytest.mark.parametrize("endpoint", ["embed", "rerank"])
async def test_paper_endpoint_429_is_not_retried(endpoint):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(429, headers={"Retry-After": "17"})

    client = make_client(handler)
    with pytest.raises(RateLimitExceeded) as caught:
        if endpoint == "embed":
            await client.embed("text", "RETRIEVAL_QUERY")
        else:
            await client.rerank("q", ["text"])
    assert caught.value.status == 429
    assert caught.value.retry_after == 17
    assert len(requests) == 1


@pytest.mark.parametrize("endpoint, payload", [
    ("embed", {}), ("embed", {"data": []}), ("embed", {"data": [None]}),
    ("embed", {"data": [{"embedding": "private-text"}]}),
    ("embed", {"data": [{"embedding": [True]}]}),
    ("embed", {"data": [{"embedding": []}]}),
    ("rerank", {}), ("rerank", {"data": []}),
    ("rerank", {"data": [{"result": {}}]}),
    ("rerank", {"data": [{"result": [{"document_id": "0"}]}]}),
    ("rerank", {"data": [{"result": [{"document_id": -1}]}]}),
    ("rerank", {"data": [{"result": [{"document_id": 1}]}]}),
    ("rerank", {"data": [{"result": [{"document_id": True}]}]}),
])
async def test_paper_endpoint_unexpected_shape_is_sarashina_error(endpoint, payload):
    client = make_client(lambda request: httpx.Response(200, json=payload))
    with pytest.raises(SarashinaError):
        if endpoint == "embed":
            await client.embed("text", "RETRIEVAL_QUERY")
        else:
            await client.rerank("q", ["text"])


@pytest.mark.parametrize("endpoint", ["embed", "rerank"])
async def test_paper_endpoint_bad_json_is_sarashina_error(endpoint):
    client = make_client(lambda request: httpx.Response(200, text="not json"))
    with pytest.raises(SarashinaError):
        if endpoint == "embed":
            await client.embed("text", "RETRIEVAL_QUERY")
        else:
            await client.rerank("q", ["text"])


async def test_embed_waits_at_default_55_rpm_and_uses_no_chat_or_translate_slots(monkeypatch):
    now = [0.0]
    sleeps = []

    async def sleep(delay):
        sleeps.append(delay)
        now[0] += delay

    monkeypatch.setattr("sarashina_bot.sarashina.asyncio.sleep", sleep)

    def handler(request):
        if request.url.path.endswith("/embeddings"):
            return httpx.Response(200, json={"data": [{"embedding": [0.1]}]})
        return by_path(request)

    client = make_client(handler, rpm=1, clock=lambda: now[0])
    for _ in range(55):
        await client.embed("text", "RETRIEVAL_QUERY")
    assert sleeps == []
    await client.chat([{"role": "user", "content": "text"}])
    await client.translate("text", "en", "ja")
    await client.embed("text", "RETRIEVAL_QUERY")
    assert sleeps == [60]


async def test_rerank_does_not_wait_or_use_chat_or_translate_slots():
    def handler(request):
        if request.url.path.endswith("/rerank"):
            return httpx.Response(200, json={"data": [{"result": []}]})
        return by_path(request)

    client = make_client(handler, rpm=1, clock=lambda: 0.0)
    await client.chat([{"role": "user", "content": "text"}])
    await client.translate("text", "en", "ja")
    for _ in range(100):
        assert await client.rerank("q", ["text"]) == []


async def test_wait_does_not_hold_lock_or_change_acquire_fail_fast(monkeypatch):
    import asyncio
    from sarashina_bot.sarashina import _RateLimiter

    now = [0.0]
    started = asyncio.Event()
    release = asyncio.Event()

    async def sleep(delay):
        started.set()
        await release.wait()
        now[0] += delay

    monkeypatch.setattr("sarashina_bot.sarashina.asyncio.sleep", sleep)
    limiter = _RateLimiter(1, clock=lambda: now[0])
    await limiter.acquire()
    waiter = asyncio.create_task(limiter.wait())
    try:
        await asyncio.wait_for(started.wait(), timeout=1)
        with pytest.raises(RateLimitExceeded):
            await asyncio.wait_for(limiter.acquire(), timeout=1)
    finally:
        release.set()
        await asyncio.wait_for(waiter, timeout=1)


async def test_concurrent_waiters_do_not_exceed_rpm(monkeypatch):
    import asyncio
    from sarashina_bot.sarashina import _RateLimiter

    now = [0.0]
    sleepers = []
    original_sleep = asyncio.sleep

    async def sleep(delay):
        assert delay > 0
        wake = asyncio.get_running_loop().create_future()
        sleepers.append(wake)
        await wake

    monkeypatch.setattr("sarashina_bot.sarashina.asyncio.sleep", sleep)
    limiter = _RateLimiter(2, clock=lambda: now[0])
    granted = []

    async def waiter():
        await limiter.wait()
        granted.append(now[0])

    tasks = [asyncio.create_task(waiter()) for _ in range(7)]
    try:
        await original_sleep(0)
        assert granted == [0.0, 0.0]
        assert len(sleepers) == 5
        for deadline, waiting in [(60.0, 3), (120.0, 1), (180.0, 0)]:
            now[0] = deadline
            waking = sleepers[:]
            sleepers.clear()
            for wake in waking:
                wake.set_result(None)
            await original_sleep(0)
            assert len(sleepers) == waiting
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=1)
        assert granted == [0.0, 0.0, 60.0, 60.0, 120.0, 120.0, 180.0]
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.parametrize("endpoint", ["embed", "rerank"])
async def test_paper_endpoint_transport_error_is_sarashina_error(endpoint):
    def handler(request):
        raise httpx.ConnectError("offline")

    client = make_client(handler)
    with pytest.raises(SarashinaError):
        if endpoint == "embed":
            await client.embed("text", "RETRIEVAL_QUERY")
        else:
            await client.rerank("q", ["text"])
