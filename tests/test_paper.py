import asyncio

import httpx
import pytest
import pymupdf

from sarashina_bot.paper import (
    MAX_PARAGRAPH,
    PaperError,
    PaperIndex,
    Paragraph,
    answer,
    build_index,
    download_pdf,
    extract_paragraphs,
    find_arxiv_id,
    format_paragraph,
    merge_blocks,
    parse_paragraph,
    retrieve,
)


@pytest.mark.parametrize("url, expected", [
    ("read <https://arxiv.org/abs/1706.03762>", "1706.03762"),
    ("http://www.arxiv.org/pdf/1706.03762v7.pdf", "1706.03762v7"),
    ("https://export.arxiv.org/pdf/hep-th/9901001v2", "hep-th/9901001v2"),
    ("https://arxiv.org/abs/math.GT/0309136", "math.GT/0309136"),
    ("https://arxiv.org/abs/1234.5678?context=cs", "1234.5678"),
    ("https://evil.example/arxiv.org/abs/1706.03762", None),
    ("https://notarxiv.org/abs/1706.03762", None),
    ("https://arxiv.org.evil.example/pdf/1706.03762", None),
    ("https://arxiv.org/abs/1706.03762/../../secret", None),
    ("https://arxiv.org/pdf/1706.03762%2Fsecret", None),
    ("https://arxiv.org/abs/1706.03762evil", None),
    ("https://arxiv.org/abs/１７０６.０３７６２", None),
    ("ftp://arxiv.org/abs/1706.03762", None),
    ("1706.03762", None),
])
def test_find_arxiv_id(url, expected):
    assert find_arxiv_id(url) == expected


def test_merge_fragments_and_drop_page_numbers_and_short_leftovers():
    fragments = ["word"] * 300
    paragraphs = merge_blocks([["1", *fragments, "23"], ["x"]])
    assert 2 <= len(paragraphs) <= 3
    assert all(len(p.text) >= 20 for p in paragraphs)
    assert " ".join(p.text for p in paragraphs).split() == fragments
    assert all(p.page == 1 for p in paragraphs)


def test_merge_breaks_before_heading_and_at_page_boundary():
    paragraphs = merge_blocks([
        ["An introductory paragraph with some text.", "2.1 Methods", "The methods follow."],
        ["This belongs to the second page."],
    ])
    assert paragraphs == [
        Paragraph(1, "An introductory paragraph with some text."),
        Paragraph(1, "2.1 Methods The methods follow."),
        Paragraph(2, "This belongs to the second page."),
    ]


def test_merge_splits_oversize_at_whitespace_and_respects_cap():
    words = ["oversized"] * 500
    paragraphs = merge_blocks([[" ".join(words)], ["x" * (MAX_PARAGRAPH * 2 + 30)]])
    assert all(0 < len(p.text) <= MAX_PARAGRAPH for p in paragraphs)
    assert " ".join(p.text for p in paragraphs if p.page == 1).split() == words
    assert "".join(p.text for p in paragraphs if p.page == 2) == "x" * (MAX_PARAGRAPH * 2 + 30)


def test_format_parse_round_trip():
    paragraph = Paragraph(3, "First line.\nSecond line.")
    assert format_paragraph(paragraph) == "p.3\nFirst line.\nSecond line."
    assert parse_paragraph(format_paragraph(paragraph)) == paragraph.text


@pytest.mark.parametrize("content", ["hello", "p.0\nbody", "p.3 body", "p.3\n", "p.3\n  ", " p.3\nbody", "p.03\nbody"])
def test_parse_rejects_non_paragraph(content):
    assert parse_paragraph(content) is None


def test_extract_tiny_pdf_in_memory():
    with pymupdf.open() as doc:
        page = doc.new_page()
        page.insert_text((72, 72), "A small generated paper with enough text for a paragraph.")
        pdf = doc.tobytes()
    assert extract_paragraphs(pdf) == [Paragraph(1, "A small generated paper with enough text for a paragraph.")]


def test_extract_blank_pdf_raises():
    with pymupdf.open() as doc:
        doc.new_page()
        pdf = doc.tobytes()
    with pytest.raises(PaperError):
        extract_paragraphs(pdf)


async def test_download_follows_redirect_and_sets_timeout_and_user_agent():
    requests = []

    def handler(request):
        requests.append(request)
        if request.url.host == "arxiv.org":
            return httpx.Response(302, headers={"location": "https://export.arxiv.org/paper.pdf"})
        return httpx.Response(200, content=b"%PDF-example")

    assert await download_pdf("1706.03762v2", transport=httpx.MockTransport(handler)) == b"%PDF-example"
    assert str(requests[0].url) == "https://arxiv.org/pdf/1706.03762v2"
    assert len(requests) == 2
    assert requests[0].headers["user-agent"]
    assert set(requests[0].extensions["timeout"].values()) == {60.0}


async def test_download_404_raises():
    with pytest.raises(PaperError, match="404"):
        await download_pdf("1706.03762", transport=httpx.MockTransport(lambda request: httpx.Response(404)))


class LargePDFStream(httpx.AsyncByteStream):
    def __init__(self):
        self.reads = 0
        self.closed = False

    async def __aiter__(self):
        for _ in range(60):
            self.reads += 1
            yield b"x" * (1024 * 1024)

    async def aclose(self):
        self.closed = True


async def test_download_stops_stream_after_50_mib():
    stream = LargePDFStream()
    transport = httpx.MockTransport(lambda request: httpx.Response(200, stream=stream))
    with pytest.raises(PaperError, match="50 MiB"):
        await download_pdf("1706.03762", transport=transport)
    assert stream.reads == 51
    assert stream.closed


async def test_download_transport_error_becomes_paper_error():
    def handler(request):
        raise httpx.ConnectError("offline")

    with pytest.raises(PaperError):
        await download_pdf("1706.03762", transport=httpx.MockTransport(handler))


async def test_build_index_limits_concurrency_and_preserves_alignment():
    gate = asyncio.Event()
    first_four_started = asyncio.Event()
    active = 0
    peak = 0
    completed = []

    class Embedder:
        async def embed(self, text, task_type):
            nonlocal active, peak
            assert task_type == "RETRIEVAL_DOCUMENT"
            active += 1
            peak = max(peak, active)
            if active == 4:
                first_four_started.set()
            if text == "0":
                await gate.wait()
            else:
                await asyncio.sleep(0)
            active -= 1
            completed.append(int(text))
            return [float(text)]

    paragraphs = [Paragraph(1, str(i)) for i in range(10)]
    task = asyncio.create_task(build_index(Embedder(), paragraphs))
    await asyncio.wait_for(first_four_started.wait(), timeout=1)
    await asyncio.sleep(0)
    assert completed and completed[0] != 0
    gate.set()
    index = await task
    assert peak == 4
    assert index.paragraphs == paragraphs
    assert index.vectors == [[float(i)] for i in range(10)]


async def test_build_index_propagates_failure():
    class Embedder:
        async def embed(self, text, task_type):
            raise RuntimeError("embedding failed")

    with pytest.raises(RuntimeError, match="embedding failed"):
        await build_index(Embedder(), [Paragraph(1, "text")])


async def test_retrieve_maps_rerank_candidate_ids_back_to_paper():
    class Retriever:
        async def embed(self, text, task_type):
            assert (text, task_type) == ("query", "RETRIEVAL_QUERY")
            return [1.0, 0.0]

        async def rerank(self, query, documents):
            assert query == "query"
            assert documents == [str(i) for i in range(29, 9, -1)]
            return [2, 0, 19, 1, 3, 4]

    index = PaperIndex(
        [Paragraph(i + 1, str(i)) for i in range(30)],
        [[float(i), 1.0] for i in range(30)],
    )
    assert await retrieve(Retriever(), index, "query") == [27, 29, 10, 28, 26]


async def test_retrieve_handles_zero_vector():
    class Retriever:
        async def embed(self, text, task_type):
            return [1.0, 0.0]

        async def rerank(self, query, documents):
            assert documents == ["relevant", "zero"]
            return [0]

    index = PaperIndex([Paragraph(1, "zero"), Paragraph(2, "relevant")], [[0.0, 0.0], [1.0, 0.0]])
    assert await retrieve(Retriever(), index, "q") == [1]


async def test_answer_is_stateless_and_uses_original_question_and_sources():
    class Answerer:
        def __init__(self):
            self.calls = []

        async def translate(self, text, source, target):
            assert (text, source, target) == ("何が分かる？", "ja", "en")
            return "What can we learn?"

        async def embed(self, text, task_type):
            assert (text, task_type) == ("What can we learn?", "RETRIEVAL_QUERY")
            return [1.0]

        async def rerank(self, query, documents):
            assert query == "What can we learn?"
            return [0, 1, 2]

        async def chat(self, messages, **kwargs):
            assert kwargs == {}  # No tools or history.
            self.calls.append(messages)
            return {"content": "結果です [p.3]。"}

    client = Answerer()
    index = PaperIndex([Paragraph(7, "excerpt seven"), Paragraph(3, "excerpt three"), Paragraph(7, "more seven")], [[1.0]] * 3)
    for _ in range(2):
        assert await answer(client, index, "何が分かる？") == "結果です [p.3]。\n\n参考: p.3, p.7"
    for messages in client.calls:
        assert [m["role"] for m in messages] == ["system", "system", "user"]
        instructions, context, question = [m["content"] for m in messages]
        assert all(word in instructions for word in ["抜粋", "日本語", "[p.", "分からない", "データ", "指示"])
        assert "[p.7] excerpt seven" in context
        assert "[p.3] excerpt three" in context
        assert "[p.7] more seven" in context
        assert question == "何が分かる？"
    assert client.calls[0] is not client.calls[1]


@pytest.mark.parametrize("finish_reason", ["length", "stop", None])
async def test_empty_answer_matches_mention_answer_without_sources(finish_reason):
    from sarashina_bot.bot import _answer

    response = {"content": "", "finish_reason": finish_reason}

    class Answerer:
        async def translate(self, text, source, target):
            return "question"

        async def chat(self, messages):
            return response

    assert await answer(Answerer(), PaperIndex([], []), "質問") == _answer(response)


async def test_build_index_429_honours_retry_after_and_retries_same_embed(monkeypatch):
    from unittest.mock import AsyncMock
    from sarashina_bot.sarashina import RateLimitExceeded

    sleep = AsyncMock()
    monkeypatch.setattr("sarashina_bot.paper.asyncio.sleep", sleep)
    calls = []

    class Embedder:
        async def embed(self, text, task_type):
            calls.append((text, task_type))
            if len(calls) < 3:
                raise RateLimitExceeded("rate limited", retry_after=len(calls) * 7, status=429)
            return [0.5]

    index = await build_index(Embedder(), [Paragraph(1, "paragraph")])
    assert index.vectors == [[0.5]]
    assert calls == [("paragraph", "RETRIEVAL_DOCUMENT")] * 3
    assert [call.args for call in sleep.await_args_list] == [(7,), (14,)]


async def test_build_index_429_propagates_after_three_attempts(monkeypatch):
    from unittest.mock import AsyncMock
    from sarashina_bot.sarashina import RateLimitExceeded

    sleep = AsyncMock()
    monkeypatch.setattr("sarashina_bot.paper.asyncio.sleep", sleep)
    failure = RateLimitExceeded("rate limited", retry_after=17, status=429)
    embed = AsyncMock(side_effect=failure)
    from types import SimpleNamespace
    with pytest.raises(RateLimitExceeded) as caught:
        await build_index(SimpleNamespace(embed=embed), [Paragraph(1, "paragraph")])
    assert caught.value is failure
    assert embed.await_count == 3
    assert [call.args for call in sleep.await_args_list] == [(17,), (17,)]


async def test_build_index_does_not_retry_local_rate_limit(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from sarashina_bot.sarashina import RateLimitExceeded

    sleep = AsyncMock()
    monkeypatch.setattr("sarashina_bot.paper.asyncio.sleep", sleep)
    embed = AsyncMock(side_effect=RateLimitExceeded("local limit", retry_after=17))
    with pytest.raises(RateLimitExceeded):
        await build_index(SimpleNamespace(embed=embed), [Paragraph(1, "paragraph")])
    assert embed.await_count == 1
    sleep.assert_not_awaited()
