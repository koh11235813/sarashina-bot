import json

import httpx
import pytest

from sarashina_bot.exa import ExaClient, ExaError


def make_client(handler):
    return ExaClient("exa-key", transport=httpx.MockTransport(handler))


def results(*items):
    return httpx.Response(200, json={"results": list(items)})


async def test_request_body_and_api_key_header():
    captured = {}

    async def handler(request):
        captured["body"] = json.loads(request.content)
        captured["key"] = request.headers.get("x-api-key")
        return results()

    client = make_client(handler)
    await client.search("sarashina", num_results=3)

    assert set(captured["body"]) == {"query", "numResults", "contents"}
    assert captured["body"]["query"] == "sarashina"
    assert captured["body"]["numResults"] == 3
    assert captured["key"] == "exa-key"


async def test_formats_numbered_results():
    client = make_client(
        lambda request: results(
            {"title": "A", "url": "https://a.example", "highlights": ["a1", "a2"]},
            {"title": "B", "url": "https://b.example", "highlights": ["b1"]},
        )
    )

    text = await client.search("q")

    assert text == (
        "[1] A\nhttps://a.example\na1\na2\n\n"
        "[2] B\nhttps://b.example\nb1"
    )


async def test_non_2xx_raises():
    client = make_client(lambda request: httpx.Response(500, text="boom"))
    with pytest.raises(ExaError):
        await client.search("q")


async def test_non_json_200_raises():
    client = make_client(lambda request: httpx.Response(200, text="<html>"))
    with pytest.raises(ExaError):
        await client.search("q")


async def test_missing_results_key_raises():
    client = make_client(lambda request: httpx.Response(200, json={"oops": 1}))
    with pytest.raises(ExaError):
        await client.search("q")
