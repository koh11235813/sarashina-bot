import pytest

from sarashina_bot.config import Config


@pytest.fixture(autouse=True)
def required_env(monkeypatch):
    for name in ("DISCORD_TOKEN", "SARASHINA_API_KEY", "EXA_API_KEY"):
        monkeypatch.setenv(name, "x")
    for name in (
        "SARASHINA_MODEL",
        "SARASHINA_MODELS",
        "DISCORD_ALLOWED_USERS",
        "MAX_TOOL_ROUNDS",
        "RATE_LIMIT_RPM",
    ):
        monkeypatch.delenv(name, raising=False)


def test_defaults_offer_only_the_default_model():
    config = Config.from_env()
    assert config.model == "sarashina3-mini"
    assert config.models == ("sarashina3-mini",)
    assert config.allowed_user_ids == frozenset()
    assert config.rate_limit_rpm == 30


def test_models_and_allowed_users_are_comma_separated(monkeypatch):
    monkeypatch.setenv("SARASHINA_MODELS", "sarashina3-mini, sarashina3.2-mini-thinking")
    monkeypatch.setenv("DISCORD_ALLOWED_USERS", "123, 456,")

    config = Config.from_env()

    assert config.models == ("sarashina3-mini", "sarashina3.2-mini-thinking")
    assert config.allowed_user_ids == frozenset({123, 456})


def test_default_model_must_be_offered(monkeypatch):
    monkeypatch.setenv("SARASHINA_MODELS", "sarashina3.2-mini-thinking")
    with pytest.raises(RuntimeError):
        Config.from_env()


def test_tool_rounds_must_fit_rate_limit(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_RPM", "5")
    monkeypatch.setenv("MAX_TOOL_ROUNDS", "5")
    with pytest.raises(RuntimeError):
        Config.from_env()
