"""Command runner for the sandbox Pod, which holds no secrets.

The bot posts ``{"command": "..."}`` to ``/run`` and gets back the exit code and
the combined stdout/stderr. Only the bot Pod can reach this server (NetworkPolicy).
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import threading
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

log = logging.getLogger(__name__)

PORT = 8080
WORKDIR = "/work"
TIMEOUT = 30.0
OUTPUT_LIMIT = 4000
_READ_CHUNK = 65536


@dataclass
class RunResult:
    exit_code: int
    timed_out: bool
    output: str


def run_command(
    command: str,
    *,
    cwd: str = WORKDIR,
    timeout: float = TIMEOUT,
    limit: int = OUTPUT_LIMIT,
) -> RunResult:
    # The Pod env is already secret-free; this also keeps Kubernetes service env vars out.
    env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": cwd,
        "LANG": "C.UTF-8",
        "TMPDIR": "/tmp",
    }
    # A new session lets the timeout kill background children too, not just bash.
    proc = subprocess.Popen(
        ["bash", "-c", command],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=cwd,
        env=env,
        start_new_session=True,
    )
    timed_out = threading.Event()

    def kill() -> None:
        timed_out.set()
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    timer = threading.Timer(timeout, kill)
    timer.start()
    try:
        # Read to EOF but keep only the head, so `yes` cannot exhaust memory.
        kept = bytearray()
        total = 0
        while chunk := proc.stdout.read(_READ_CHUNK):
            total += len(chunk)
            if len(kept) < limit:
                kept += chunk[: limit - len(kept)]
        exit_code = proc.wait()
    finally:
        timer.cancel()
        proc.stdout.close()

    output = kept.decode("utf-8", errors="replace")
    if total > len(kept):
        output += f"\n（出力が長いので先頭 {len(kept)} バイトまでにしました。全体は {total} バイト）"
    return RunResult(exit_code=exit_code, timed_out=timed_out.is_set(), output=output)


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        if self.path != "/run":
            self._reply(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            command = json.loads(self.rfile.read(length))["command"]
        except (ValueError, KeyError, TypeError):
            self._reply(400, {"error": "body must be {\"command\": str}"})
            return
        if not isinstance(command, str) or not command:
            self._reply(400, {"error": "body must be {\"command\": str}"})
            return
        log.info("run: %s", command)
        try:
            result = run_command(command)
        except (OSError, ValueError) as exc:
            # Without a reply, socketserver just drops the connection and the bot never learns why.
            log.warning("run failed: %s", exc)
            self._reply(500, {"error": str(exc)})
            return
        self._reply(200, asdict(result))

    def _reply(self, status: int, body: dict) -> None:
        data = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args) -> None:
        log.debug(format, *args)


def main() -> None:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
    server = ThreadingHTTPServer(("", PORT), _Handler)
    log.info("sandbox listening on :%d", PORT)
    server.serve_forever()


if __name__ == "__main__":
    main()
