from __future__ import annotations

import logging
from dataclasses import dataclass

import discord

from .config import Config
from .sarashina import SarashinaClient

log = logging.getLogger(__name__)

MAX_MESSAGE_LENGTH = 2000


@dataclass
class HistoryMessage:
    id: int
    name: str
    content: str


def build_messages(
    history: list[HistoryMessage],
    trigger: HistoryMessage,
    system_prompt: str,
) -> list[dict[str, str]]:
    """Build the chat payload from oldest-first history plus the trigger message."""
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    seen: set[int] = set()
    for message in history:
        if message.id == trigger.id or message.id in seen:
            continue
        seen.add(message.id)
        if not message.content:
            continue
        messages.append(
            {"role": "user", "content": f"{message.name}: {message.content}"}
        )
    if trigger.content:
        messages.append({"role": "user", "content": f"{trigger.name}: {trigger.content}"})
    return messages


def split_message(text: str, limit: int = MAX_MESSAGE_LENGTH) -> list[str]:
    if not text:
        return []
    chunks: list[str] = []
    remaining = text
    while len(remaining) > limit:
        split_at = remaining.rfind("\n", 1, limit + 1)
        if split_at == -1:
            split_at = limit
        chunks.append(remaining[:split_at])
        remaining = remaining[split_at:]
        if remaining.startswith("\n"):
            remaining = remaining[1:]
    if remaining:
        chunks.append(remaining)
    return chunks


class SarashinaBot(discord.Client):
    def __init__(self, config: Config, sarashina: SarashinaClient) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)
        self._config = config
        self._sarashina = sarashina

    async def on_message(self, message: discord.Message) -> None:
        if message.author.id == self.user.id:
            return
        if not self.user.mentioned_in(message):
            return

        try:
            history = [
                m
                async for m in message.channel.history(
                    limit=self._config.context_message_count
                )
            ]
            # history() yields newest-first; the model needs oldest-first.
            history.reverse()
            records = [
                HistoryMessage(
                    id=m.id, name=m.author.display_name, content=m.clean_content
                )
                for m in history
            ]
            trigger = HistoryMessage(
                id=message.id,
                name=message.author.display_name,
                content=message.clean_content,
            )
            payload = build_messages(records, trigger, self._config.system_prompt)

            async with message.channel.typing():
                reply = await self._sarashina.chat(payload)
            chunks = split_message(reply)
            if not chunks:
                return
            allowed = discord.AllowedMentions.none()
            await message.reply(chunks[0], allowed_mentions=allowed)
            for chunk in chunks[1:]:
                await message.channel.send(chunk, allowed_mentions=allowed)
        except Exception:
            log.exception("failed to generate reply")
            await message.reply(
                "ごめんなさい、返信に失敗しました。",
                allowed_mentions=discord.AllowedMentions.none(),
            )
