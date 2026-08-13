from __future__ import annotations

import time
from types import SimpleNamespace

from src.compressor.summarizer import LLMSummarizer
from src.runtime import CancellationToken


def test_llm_compressor_forwards_provider_deadline() -> None:
    class _Policy:
        def __init__(self) -> None:
            self.timeout = None

        def __call__(self, _messages):
            raise AssertionError("bounded compression must use call_with_timeout")

        def call_with_timeout(self, _messages, timeout_seconds):
            self.timeout = timeout_seconds
            return SimpleNamespace(content="bounded summary")

    policy = _Policy()
    summarizer = LLMSummarizer(policy)
    result = summarizer.summarize_document(
        "long input " * 200,
        max_length=100,
        request_deadline_monotonic=time.monotonic() + 3,
    )

    assert result == "bounded summary"
    assert policy.timeout is not None
    assert 0.25 <= policy.timeout <= 3


def test_llm_compressor_does_not_call_provider_after_cancellation() -> None:
    class _Policy:
        def __call__(self, _messages):
            raise AssertionError("cancelled compression must not call the model")

    token = CancellationToken()
    token.request("stop")
    source = "long input " * 200
    result = LLMSummarizer(_Policy()).summarize_document(
        source,
        max_length=100,
        cancellation_token=token,
    )

    assert result.startswith(source[:100])
    assert result.endswith("[TRUNCATED]")
