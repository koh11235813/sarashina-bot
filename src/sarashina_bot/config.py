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
    model: str = "sarashina3-mini"
    context_message_count: int = 10
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    rate_limit_rpm: int = 5
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "Config":
        log_level = os.getenv("LOG_LEVEL", "INFO")
        logging.basicConfig(level=log_level.upper())
        return cls(
            discord_token=_required("DISCORD_TOKEN"),
            api_key=_required("SARASHINA_API_KEY"),
            model=os.getenv("SARASHINA_MODEL", "sarashina3-mini"),
            context_message_count=int(os.getenv("CONTEXT_MESSAGE_COUNT", "10")),
            system_prompt=os.getenv("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT),
            rate_limit_rpm=int(os.getenv("RATE_LIMIT_RPM", "5")),
            log_level=log_level,
        )


def _required(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} is required")
    return value
