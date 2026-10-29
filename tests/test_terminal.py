import json

import httpx
import pytest

from sarashina_bot.terminal import TerminalClient, TerminalError


def make_client(handler):
    return TerminalClient("http://sandbox:8080", transport=httpx.MockTransport(handler))


def result(exit_code=0, timed_out=False, output=""):
    return httpx.Response(
        200, json={"exit_code": exit_code, "timed_out": timed_out, "output": output}
    )


async def test_posts_command_to_run():
    captured = {}

    async def handler(request):
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return result()

    await make_client(handler).run("ls -la")

    assert captured["url"] == "http://sandbox:8080/run"
    assert captured["body"] == {"command": "ls -la"}


async def test_formats_exit_code_and_output():
    client = make_client(lambda request: result(exit_code=2, output="nope\n"))
    assert await client.run("false") == "終了コード 2\nnope\n"


async def test_formats_timeout():
    client = make_client(lambda request: result(exit_code=-9, timed_out=True, output="x"))
    text = await client.run("sleep 99")
    assert text.startswith("タイムアウト")
    assert text.endswith("x")


async def test_non_2xx_raises():
    client = make_client(lambda request: httpx.Response(400, json={"error": "bad"}))
    with pytest.raises(TerminalError):
        await client.run("ls")


async def test_unexpected_body_raises():
    client = make_client(lambda request: httpx.Response(200, json={"oops": 1}))
    with pytest.raises(TerminalError):
        await client.run("ls")


async def test_connection_error_raises():
    def handler(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(TerminalError):
        await make_client(handler).run("ls")
