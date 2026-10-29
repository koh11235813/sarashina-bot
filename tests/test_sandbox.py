import json
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from sarashina_bot import sandbox
from sarashina_bot.sandbox import run_command


def test_captures_stdout_stderr_and_exit_code(tmp_path):
    result = run_command("echo out; echo err >&2; exit 3", cwd=str(tmp_path))

    assert result.exit_code == 3
    assert not result.timed_out
    assert result.output == "out\nerr\n"


def test_runs_in_workdir(tmp_path):
    result = run_command("pwd", cwd=str(tmp_path))
    assert result.output.strip() == str(tmp_path)


def test_env_has_no_inherited_variables(tmp_path, monkeypatch):
    monkeypatch.setenv("DISCORD_TOKEN", "leak")
    result = run_command("env", cwd=str(tmp_path))
    assert "leak" not in result.output


def test_timeout_kills_background_children(tmp_path):
    start = time.monotonic()
    result = run_command("sleep 30 & sleep 30", cwd=str(tmp_path), timeout=0.5)

    assert result.timed_out
    assert time.monotonic() - start < 10


def test_long_output_is_truncated(tmp_path):
    result = run_command("head -c 100000 /dev/zero | tr '\\0' a", cwd=str(tmp_path), limit=10)

    assert result.output.startswith("a" * 10 + "\n")
    assert "100000" in result.output


def test_invalid_utf8_is_replaced(tmp_path):
    result = run_command("printf '\\xff'", cwd=str(tmp_path))
    assert result.output == "�"


@pytest.fixture
def server_url(tmp_path, monkeypatch):
    # /work does not exist outside the sandbox Pod.
    monkeypatch.setattr(
        sandbox, "run_command", lambda command: run_command(command, cwd=str(tmp_path))
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), sandbox._Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def post(url, body):
    request = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read())


def test_http_run(server_url):
    body = post(f"{server_url}/run", {"command": "echo hi"})
    assert body == {"exit_code": 0, "timed_out": False, "output": "hi\n"}


@pytest.mark.parametrize("body", [{}, {"command": ""}, {"command": 1}])
def test_http_rejects_bad_body(server_url, body):
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        post(f"{server_url}/run", body)
    assert excinfo.value.code == 400


def test_http_unknown_path(server_url):
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        post(f"{server_url}/other", {"command": "echo hi"})
    assert excinfo.value.code == 404
