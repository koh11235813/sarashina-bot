from sarashina_bot.bot import HistoryMessage, build_messages, split_message


def rec(id, name, content):
    return HistoryMessage(id=id, name=name, content=content)


def test_system_prompt_is_first():
    out = build_messages([rec(1, "a", "one")], rec(2, "b", "two"), "sys")
    assert out[0] == {"role": "system", "content": "sys"}


def test_ordering_is_preserved_and_trigger_last():
    history = [rec(1, "a", "one"), rec(2, "b", "two")]
    out = build_messages(history, rec(2, "b", "two"), "sys")
    assert [m["content"] for m in out[1:]] == ["a: one", "b: two"]


def test_trigger_appended_when_absent_from_history():
    history = [rec(1, "a", "one")]
    out = build_messages(history, rec(9, "z", "hey"), "sys")
    assert out[-1] == {"role": "user", "content": "z: hey"}


def test_duplicate_ids_are_deduped():
    history = [rec(1, "a", "one"), rec(1, "a", "one"), rec(2, "b", "two")]
    out = build_messages(history, rec(3, "c", "three"), "sys")
    assert [m["content"] for m in out[1:]] == ["a: one", "b: two", "c: three"]


def test_empty_content_is_skipped():
    history = [rec(1, "a", ""), rec(2, "b", "two")]
    out = build_messages(history, rec(3, "c", "three"), "sys")
    assert [m["content"] for m in out[1:]] == ["b: two", "c: three"]


def test_empty_trigger_is_not_appended():
    out = build_messages([rec(1, "a", "one")], rec(2, "b", ""), "sys")
    assert [m["content"] for m in out[1:]] == ["a: one"]


def test_split_short_text():
    assert split_message("hello") == ["hello"]


def test_split_exactly_2000_chars():
    text = "a" * 2000
    assert split_message(text) == [text]


def test_split_2001_chars():
    text = "a" * 2001
    chunks = split_message(text)
    assert [len(c) for c in chunks] == [2000, 1]
    assert "".join(chunks) == text


def test_split_prefers_newlines():
    text = "a" * 10 + "\n" + "b" * 10
    chunks = split_message(text, limit=11)
    assert len(chunks) == 2
    assert all(len(c) <= 11 for c in chunks)
    assert "".join(chunks) == "a" * 10 + "b" * 10


def test_split_long_without_newlines():
    text = "a" * 4500
    chunks = split_message(text)
    assert all(0 < len(c) <= 2000 for c in chunks)
    assert "".join(chunks) == text


def test_split_long_with_newlines():
    text = ("line\n" * 1000)
    chunks = split_message(text)
    assert all(0 < len(c) <= 2000 for c in chunks)
    assert len(chunks) > 1
    # Only the newline at each split point is consumed.
    assert "\n".join(chunks) == text


def test_split_empty_text():
    assert split_message("") == []
