from __future__ import annotations

from .bot import SarashinaBot
from .config import Config
from .exa import ExaClient
from .sarashina import SarashinaClient
from .terminal import TerminalClient


def main() -> None:
    config = Config.from_env()
    sarashina = SarashinaClient(
        config.api_key, config.model, rpm=config.rate_limit_rpm
    )
    exa = ExaClient(config.exa_api_key)
    terminal = TerminalClient(config.sandbox_url)
    bot = SarashinaBot(config, sarashina, exa, terminal)
    # Config.from_env already configured the root logger; discord.py's own handler would duplicate every line.
    bot.run(config.discord_token, log_handler=None)


if __name__ == "__main__":
    main()
