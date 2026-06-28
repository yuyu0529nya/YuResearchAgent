"""
tests/unit/test_model_router.py
src/models/model_router.py 的后端配置逻辑测试（坐实"4 后端热切换"）。

只测 _load_backend_config / _is_backend_configured 的纯配置解析（不实例化
VLLMPolicy、不发起网络请求）。用 monkeypatch 控制环境变量。
"""
import pytest

from src.models.model_router import ModelRouter


def test_deepseek_defaults(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
    cfg = ModelRouter._load_backend_config("deepseek")
    assert cfg["model_name"] == "deepseek-chat"
    assert cfg["base_url"] == "https://api.deepseek.com/v1"
    assert cfg["api_key"] == "sk-test"


def test_openai_defaults(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    cfg = ModelRouter._load_backend_config("openai")
    assert cfg["model_name"] == "gpt-4o"
    assert "api.openai.com" in cfg["base_url"]


def test_mimo_defaults(monkeypatch):
    monkeypatch.setenv("MIMO_API_KEY", "k")
    monkeypatch.delenv("MIMO_BASE_URL", raising=False)
    monkeypatch.delenv("MIMO_MODEL", raising=False)
    cfg = ModelRouter._load_backend_config("mimo")
    assert cfg["model_name"] == "mimo-v2.5-pro"
    assert "xiaomimimo" in cfg["base_url"]


def test_vllm_defaults_when_base_url_present(monkeypatch):
    monkeypatch.setenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.delenv("VLLM_MODEL", raising=False)
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    cfg = ModelRouter._load_backend_config("vllm")
    assert cfg["model_name"] == "Qwen/Qwen2.5-7B-Instruct"
    assert cfg["api_key"] == "EMPTY"


def test_unconfigured_backend_raises(monkeypatch):
    monkeypatch.delenv("GHOST_API_KEY", raising=False)
    monkeypatch.delenv("GHOST_BASE_URL", raising=False)
    with pytest.raises(ValueError):
        ModelRouter._load_backend_config("ghost")


def test_optional_sampling_params_parsed(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    monkeypatch.setenv("DEEPSEEK_TEMPERATURE", "0.5")
    monkeypatch.setenv("DEEPSEEK_MAX_TOKENS", "2048")
    cfg = ModelRouter._load_backend_config("deepseek")
    assert cfg["temperature"] == 0.5
    assert cfg["max_tokens"] == 2048


def test_is_backend_configured(monkeypatch):
    monkeypatch.setenv("FOO_API_KEY", "k")
    monkeypatch.delenv("BAR_API_KEY", raising=False)
    monkeypatch.delenv("BAR_BASE_URL", raising=False)
    assert ModelRouter._is_backend_configured("foo") is True
    assert ModelRouter._is_backend_configured("bar") is False
