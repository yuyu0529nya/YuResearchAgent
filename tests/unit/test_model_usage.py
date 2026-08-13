from types import SimpleNamespace

from src.models.vllm_policy import VLLMPolicy
from src.runtime import UsageTracker


def _response(prompt_tokens: int = 10, completion_tokens: int = 4):
    message = SimpleNamespace(content="ok", tool_calls=[], reasoning_content=None)
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


def test_policy_records_provider_token_usage() -> None:
    policy = VLLMPolicy(api_key="test")
    policy.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_: _response())
        )
    )
    global_before = VLLMPolicy.global_usage_snapshot()

    result = policy([{"role": "user", "content": "hello"}])

    assert result["content"] == "ok"
    assert policy.usage_snapshot() == {
        "api_calls": 1,
        "failed_calls": 0,
        "prompt_tokens": 10,
        "completion_tokens": 4,
        "total_tokens": 14,
    }
    global_after = VLLMPolicy.global_usage_snapshot()
    assert global_after["total_tokens"] - global_before["total_tokens"] == 14


def test_policy_records_failed_calls() -> None:
    policy = VLLMPolicy(api_key="test")

    def fail(**_):
        raise RuntimeError("temporary failure")

    policy.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=fail))
    )

    result = policy([{"role": "user", "content": "hello"}])

    assert result["content"].startswith("Error:")
    assert policy.usage_snapshot()["api_calls"] == 1
    assert policy.usage_snapshot()["failed_calls"] == 1


def test_policy_applies_request_deadline_and_disables_retries() -> None:
    class _Client:
        def __init__(self) -> None:
            self.options = None
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **_: (_ for _ in ()).throw(
                        AssertionError("base client must not execute the bounded request")
                    )
                )
            )

        def with_options(self, **options):
            self.options = options
            return SimpleNamespace(
                chat=SimpleNamespace(
                    completions=SimpleNamespace(create=lambda **_: _response())
                )
            )

    policy = VLLMPolicy(api_key="test")
    client = _Client()
    policy.client = client

    result = policy.call_with_timeout(
        [{"role": "user", "content": "hello"}],
        timeout_seconds=7.5,
    )

    assert result["content"] == "ok"
    assert client.options == {"timeout": 7.5, "max_retries": 0}


def test_policy_deadline_fails_closed_when_client_cannot_enforce_it() -> None:
    called = False

    def create(**_):
        nonlocal called
        called = True
        return _response()

    policy = VLLMPolicy(api_key="test")
    policy.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    result = policy.call_with_timeout(
        [{"role": "user", "content": "hello"}],
        timeout_seconds=1,
    )

    assert result["content"].startswith("Error:")
    assert "cannot enforce" in result["content"]
    assert called is False


def test_usage_tracker_aggregates_multiple_policy_instances() -> None:
    tracker = UsageTracker()
    first = VLLMPolicy(api_key="test", usage_tracker=tracker)
    second = VLLMPolicy(api_key="test", usage_tracker=tracker)
    first.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_: _response(8, 2))
        )
    )
    second.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_: _response(5, 3))
        )
    )

    first([{"role": "user", "content": "one"}])
    second([{"role": "user", "content": "two"}])

    assert tracker.snapshot() == {
        "api_calls": 2,
        "failed_calls": 0,
        "prompt_tokens": 13,
        "completion_tokens": 5,
        "total_tokens": 18,
    }
