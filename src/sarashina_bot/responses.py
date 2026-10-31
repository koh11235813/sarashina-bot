from __future__ import annotations

_LENGTH_MESSAGE = "回答が出せませんでした。effort を下げてください。"


def answer_text(response: dict) -> str:
    # A high effort can spend max_completion_tokens on thinking and stop before any answer.
    if not response["content"] and response.get("finish_reason") == "length":
        return _LENGTH_MESSAGE
    return response["content"]
