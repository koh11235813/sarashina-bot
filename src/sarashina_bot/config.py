from __future__ import annotations

import logging
import os
from dataclasses import dataclass

DEFAULT_SYSTEM_PROMPT = (
    "あなたはSarashinaです。日本語で親しみやすく、簡潔に答えるアシスタントです。"
)


@dataclass(frozen=True)
class Config:
    discord_token: str
    api_key: str
    exa_api_key: str
    model: str = "sarashina3-mini"
    models: tuple[str, ...] = ("sarashina3-mini",)
    allowed_user_ids: frozenset[int] = frozenset()
    sandbox_url: str = "http://sandbox:8080"
    max_tool_rounds: int = 2
    context_message_count: int = 10
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    rate_limit_rpm: int = 5
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "Config":
        log_level = os.getenv("LOG_LEVEL", "INFO")
        logging.basicConfig(level=log_level.upper())
        model = os.getenv("SARASHINA_MODEL", "sarashina3-mini")
        models = tuple(_split(os.getenv("SARASHINA_MODELS", model)))
        if model not in models:
            raise RuntimeError(f"SARASHINA_MODEL {model} is not in SARASHINA_MODELS")
        max_tool_rounds = int(os.getenv("MAX_TOOL_ROUNDS", "2"))
        rate_limit_rpm = int(os.getenv("RATE_LIMIT_RPM", "5"))
        # A mention reserves max_tool_rounds + 1 calls up front, so it could never pass the limiter.
        if max_tool_rounds + 1 > rate_limit_rpm:
            raise RuntimeError("MAX_TOOL_ROUNDS + 1 must not exceed RATE_LIMIT_RPM")
        return cls(
            discord_token=_required("DISCORD_TOKEN"),
            api_key=_required("SARASHINA_API_KEY"),
            exa_api_key=_required("EXA_API_KEY"),
            model=model,
            models=models,
            allowed_user_ids=frozenset(
                int(v) for v in _split(os.getenv("DISCORD_ALLOWED_USERS", ""))
            ),
            sandbox_url=os.getenv("SANDBOX_URL", "http://sandbox:8080"),
            max_tool_rounds=max_tool_rounds,
            context_message_count=int(os.getenv("CONTEXT_MESSAGE_COUNT", "10")),
            system_prompt=os.getenv("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT),
            rate_limit_rpm=rate_limit_rpm,
            log_level=log_level,
        )


def _split(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def _required(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} is required")
    return value
