from types import SimpleNamespace

from src.models.vllm_policy import VLLMPolicy


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
