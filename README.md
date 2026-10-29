[日本語](README.ja.md)

# sarashina-bot

A Discord bot that replies to mentions using the [Sarashina](https://api.sarashina.ai/v1) chat API.

- The model can search the web through [Exa](https://exa.ai) (`web_search` tool, offered to everyone).
- Users listed in `DISCORD_ALLOWED_USERS` also get a `run_command` tool, which runs bash in a separate sandbox pod over HTTP.
- `/model [name]` lets allowed users switch between the models in `SARASHINA_MODELS` (reply is ephemeral; the switch lasts until the process restarts).

Design details and Sarashina API quirks are in [SPEC.md](SPEC.md) (Japanese).

## Architecture

One image, two processes:

| Process | Command | Role |
| --- | --- | --- |
| Bot | `python -m sarashina_bot` (image default) | Connects to Discord, calls Sarashina and Exa, forwards `run_command` to the sandbox |
| Sandbox | `python -m sarashina_bot.sandbox` | HTTP server on port 8080; `POST /run` with `{"command": str}` returns `{"exit_code", "timed_out", "output"}` |

The sandbox:

- runs as UID 10001 with `/work` as the working directory and `HOME`; `/work` and `/tmp` are the writable places
- runs each command with `bash -c` in a new session, and kills the whole process group with SIGKILL after 30 seconds
- returns the first 4,000 bytes of combined stdout and stderr
- passes only `PATH`, `HOME`, `LANG` and `TMPDIR` to the command, and holds no secrets
- has no authentication; in the deployment a NetworkPolicy only lets the bot pod reach it

Per mention the bot calls the model at most `MAX_TOOL_ROUNDS + 1` times: up to `MAX_TOOL_ROUNDS` rounds with tools, then a final call without tools to force a text answer.

### Conversation context

On a mention the bot fetches the latest `CONTEXT_MESSAGE_COUNT` messages in the channel and keeps only the mentioning user's messages (as `user`) and the bot's replies to that user, including the continuation chunks of a split reply (as `assistant`). Everyone else's messages are dropped. A leading `assistant` message is dropped so the conversation starts with `user`. The system prompt gets "the person talking to you is X" appended. Nothing is stored; history is read from Discord each time.

## Configuration

All settings are environment variables (see `src/sarashina_bot/config.py`).

| Variable | Required | Default | Meaning |
| --- | --- | --- | --- |
| `DISCORD_TOKEN` | yes | | Discord bot token |
| `SARASHINA_API_KEY` | yes | | Sarashina API key |
| `EXA_API_KEY` | yes | | Exa API key for `web_search` |
| `DISCORD_ALLOWED_USERS` | no | empty | Comma-separated Discord user IDs that get `run_command` and `/model` |
| `SARASHINA_MODEL` | no | `sarashina3-mini` | Model used at startup; must be in `SARASHINA_MODELS` |
| `SARASHINA_MODELS` | no | value of `SARASHINA_MODEL` | Comma-separated choices for `/model` |
| `SANDBOX_URL` | no | `http://sandbox:8080` | Base URL of the sandbox |
| `MAX_TOOL_ROUNDS` | no | `2` | Tool rounds per mention; `MAX_TOOL_ROUNDS + 1` must not exceed `RATE_LIMIT_RPM` |
| `RATE_LIMIT_RPM` | no | `30` | Client-side limit on Sarashina requests per minute. When it or Sarashina's own limit (429) is hit, the bot replies 「{x}秒後にもう一度送信してください。」 with the seconds to wait |
| `CONTEXT_MESSAGE_COUNT` | no | `10` | How many recent channel messages to read before filtering |
| `SYSTEM_PROMPT` | no | short built-in Japanese prompt | System prompt |
| `LOG_LEVEL` | no | `INFO` | Python log level (also read by the sandbox) |

The bot refuses to start if a required variable is missing, if `SARASHINA_MODEL` is not in `SARASHINA_MODELS`, or if `MAX_TOOL_ROUNDS + 1 > RATE_LIMIT_RPM`. The sandbox reads only `LOG_LEVEL`.

## Development

Requires [uv](https://docs.astral.sh/uv/). The Python version is pinned in `.python-version`.

```sh
uv sync
uv run pytest
```

Run the bot locally:

```sh
export DISCORD_TOKEN=... SARASHINA_API_KEY=... EXA_API_KEY=...
export DISCORD_ALLOWED_USERS=123456789012345678   # optional
export SANDBOX_URL=http://localhost:8080          # only needed for run_command
uv run python -m sarashina_bot
```

Run the sandbox locally. The working directory `/work` is hard-coded and must exist (in Kubernetes it is a mounted emptyDir); without it every request returns 500. Commands run as your own user, so prefer running the sandbox in the container:

```sh
podman run --rm -p 8080:8080 --tmpfs /work:mode=1777 \
  localhost/sarashina_bot:dev python -m sarashina_bot.sandbox
curl -s localhost:8080/run -d '{"command": "id; pwd"}'
```

## Build

```sh
podman build -t localhost/sarashina_bot:<tag> .
```

`docker build` works too. The Dockerfile uses `RUN --mount=type=cache`, so Docker needs BuildKit (the default in current versions).

CI (`.github/workflows/ci.yml`) runs the tests and `docker build .` on pull requests and pushes to `main`. It does not push images.

## Deployment

The Kubernetes manifests and the deploy script live in the private homelab repo, under [`sarashina-bot/`](https://github.com/koh11235813/homelab/tree/main/sarashina-bot). `sarashina-bot/deploy.sh <path to this clone>` builds the image and rolls it out; see that directory for the steps.
