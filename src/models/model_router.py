"""
多后端 LLM 路由器 (Model Router)

支持通过环境变量 (.env) 配置多个 LLM 后端，运行时动态切换：
  - Kimi Code / Moonshot API
  - DeepSeek API
  - 本地 vLLM
  - OpenAI / 任何 OpenAI 兼容 API

设计要点:
  1. 零源码修改切换后端：所有敏感信息（API Key / URL）都放在 .env 文件中
  2. 运行时热切换：不同模块可以用不同后端（如 Red Agent 用 cheap 模型，Solver 用 strong 模型）
  3. 向后兼容：保留 VLLMPolicy 的所有接口和行为，只扩展初始化方式

用法示例:
  >>> from src.models.model_router import ModelRouter
  >>> # 使用默认后端（.env 中 DEFAULT_LLM_BACKEND 指定）
  >>> policy = ModelRouter.create_backend()
  >>> # 显式指定后端
  >>> policy = ModelRouter.create_backend("deepseek")
  >>> # 获取所有可用后端，按场景分配
  >>> backends = ModelRouter.get_all_backends()
  >>> solver_policy = backends["deepseek"]
  >>> red_policy = backends["vllm"]  # 本地模型，攻击成本低
"""
from __future__ import annotations

import os

from ..utils.env_config import ensure_env_loaded, get_env
from .vllm_policy import VLLMPolicy


__all__ = ["ModelRouter"]

# 全局缓存，避免重复读取 .env 和创建 client
_BACKEND_CACHE: dict[str, VLLMPolicy] = {}


class ModelRouter:
    """多后端 LLM 路由器。

    所有方法都是类方法 / 静态方法，无需实例化。
    """

    @staticmethod
    def create_backend(
        backend_name: str | None = None,
        *,
        use_cache: bool = True,
        **override_kwargs,
    ) -> VLLMPolicy:
        """创建指定名称的 LLM Backend（返回 VLLMPolicy 实例）。

        Args:
            backend_name: 后端名称，对应 .env 中的前缀。
                          为 None 时使用 DEFAULT_LLM_BACKEND。
            use_cache: 是否复用同配置 policy。并发 worker 应设为 False，
                       避免共享 tools / was_truncated 等可变运行状态。
            **override_kwargs: 覆盖 .env 中任何参数（如 temperature, max_tokens）。

        Returns:
            VLLMPolicy: 配置好的策略实例，接口保持统一一致。

        Raises:
            ValueError: 找不到对应后端配置时。
        """
        ensure_env_loaded()

        name = (backend_name or get_env("DEFAULT_LLM_BACKEND", "vllm")).lower().strip()

        # 检查缓存
        cache_key = f"{name}:{hash(tuple(sorted(override_kwargs.items())))}"
        if use_cache and cache_key in _BACKEND_CACHE:
            return _BACKEND_CACHE[cache_key]

        # 根据名称读取 .env 配置
        config = ModelRouter._load_backend_config(name)
        config.update(override_kwargs)
        config = ModelRouter._normalize_backend_config(name, config)

        # 创建 VLLMPolicy 实例
        policy = VLLMPolicy(**config)
        if use_cache:
            _BACKEND_CACHE[cache_key] = policy
        return policy

    @staticmethod
    def get_all_backends(backend_names: list[str] | None = None) -> dict[str, VLLMPolicy]:
        """预加载并返回所有已配置的后端。

        Args:
            backend_names: 指定要扫描的后端名称列表。为 None 时扫描全部已知后端
                          （kimi, qwen, glm, deepseek, vllm, openai, mimo 及任何自定义前缀）。

        常用于"主模型用 DeepSeek，Red Agent 用 MiMo"的场景。
        """
        ensure_env_loaded()
        backends: dict[str, VLLMPolicy] = {}

        # 默认扫描所有已知内置后端 + 环境变量中发现的自定义后端
        if backend_names is None:
            backend_names = ["kimi", "qwen", "glm", "deepseek", "vllm", "openai", "mimo"]
            # 自动发现 .env 中其他以 _API_KEY 结尾的自定义后端
            for key in os.environ:
                if key.endswith("_API_KEY"):
                    prefix = key[:-len("_API_KEY")].lower()
                    if prefix not in backend_names:
                        backend_names.append(prefix)

        for name in backend_names:
            if ModelRouter._is_backend_configured(name):
                try:
                    backends[name] = ModelRouter.create_backend(name)
                except ValueError:
                    pass  # 配置不完整，跳过
        return backends

    @staticmethod
    def clear_cache() -> None:
        """清空后端缓存。用于配置热重载后重新初始化。"""
        _BACKEND_CACHE.clear()

    @staticmethod
    def _is_backend_configured(name: str) -> bool:
        """检查某个后端是否已在 .env 中配置。"""
        prefix = name.upper()
        return get_env(f"{prefix}_API_KEY") is not None or get_env(f"{prefix}_BASE_URL") is not None

    @staticmethod
    def _normalize_backend_config(name: str, config: dict) -> dict:
        """Apply provider constraints after environment and module overrides are merged."""
        normalized = dict(config)
        model_name = str(normalized.get("model_name", "")).lower()

        # Kimi Code K3 currently accepts only this sampling pair.
        if name == "kimi" and model_name in {"k3", "k3-256k"}:
            normalized["temperature"] = 1.0
            normalized["top_p"] = 0.95

        return normalized

    @staticmethod
    def _load_backend_config(name: str) -> dict:
        """从环境变量加载指定后端的配置字典。

        环境变量命名规范: {PREFIX}_{PARAM}
          例如: DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
        """
        prefix = name.upper()

        api_key = get_env(f"{prefix}_API_KEY")
        base_url = get_env(f"{prefix}_BASE_URL")
        model = get_env(f"{prefix}_MODEL")

        if api_key is None and base_url is None:
            raise ValueError(
                f"后端 '{name}' 未配置。请在 .env 或 .env.local 中设置 "
                f"{prefix}_API_KEY 和/或 {prefix}_BASE_URL。"
            )

        # 构建 VLLMPolicy 接受的参数字典
        config: dict = {}

        if model is not None:
            config["model_name"] = model
        if base_url is not None:
            config["base_url"] = base_url
        if api_key is not None:
            config["api_key"] = api_key

        # 可选参数
        temp = get_env(f"{prefix}_TEMPERATURE")
        if temp is not None:
            config["temperature"] = float(temp)

        max_tok = get_env(f"{prefix}_MAX_TOKENS")
        if max_tok is not None:
            config["max_tokens"] = int(max_tok)

        max_input_chars = get_env(f"{prefix}_MAX_INPUT_CHARS")
        if max_input_chars is not None:
            config["max_input_chars"] = int(max_input_chars)

        extra_body: dict = {}

        # 推理模型（GLM-4.5/4.6 等）：关闭思考以加速、并确保 content 字段非空
        if (get_env(f"{prefix}_DISABLE_THINKING") or "").lower() in ("true", "1", "yes"):
            extra_body["thinking"] = {"type": "disabled"}

        # Kimi K3 等推理模型支持 low/high/max；作为 extra_body 透传给兼容 API。
        reasoning_effort = get_env(f"{prefix}_REASONING_EFFORT")
        if reasoning_effort:
            extra_body["reasoning_effort"] = reasoning_effort.lower().strip()

        if extra_body:
            config["extra_body"] = extra_body

        # 为 vllm 提供默认值（如果用户没配 model）
        if name == "vllm" and "model_name" not in config:
            config["model_name"] = "Qwen/Qwen2.5-7B-Instruct"
        if name == "vllm" and "base_url" not in config:
            config["base_url"] = "http://localhost:8000/v1"
        if name == "vllm" and "api_key" not in config:
            config["api_key"] = "EMPTY"

        # 为 deepseek 提供默认值
        if name == "deepseek" and "model_name" not in config:
            config["model_name"] = "deepseek-chat"
        if name == "deepseek" and "base_url" not in config:
            config["base_url"] = "https://api.deepseek.com/v1"

        # 为 openai 提供默认值
        if name == "openai" and "model_name" not in config:
            config["model_name"] = "gpt-4o"
        if name == "openai" and "base_url" not in config:
            config["base_url"] = "https://api.openai.com/v1"

        # Kimi Code 会员 API（K3）
        if name == "kimi" and "model_name" not in config:
            config["model_name"] = "k3"
        if name == "kimi" and "base_url" not in config:
            config["base_url"] = "https://api.kimi.com/coding/v1"

        # 为 qwen / DashScope OpenAI 兼容接口提供默认值
        if name == "qwen" and "model_name" not in config:
            config["model_name"] = "qwen-plus"
        if name == "qwen" and "base_url" not in config:
            config["base_url"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

        # 为 zhipu GLM OpenAI 兼容接口提供默认值
        if name == "glm" and "model_name" not in config:
            config["model_name"] = "glm-4.6"
        if name == "glm" and "base_url" not in config:
            config["base_url"] = "https://open.bigmodel.cn/api/paas/v4"

        # 为 xiaomi mimo 提供默认值（官方 API，国内直连）
        # 官方平台: https://platform.xiaomimimo.com/
        # 如需走 OpenRouter 备选渠道，在 .env 中手动覆盖：
        #   MIMO_BASE_URL=https://openrouter.ai/api/v1
        #   MIMO_MODEL=xiaomi/mimo-v2.5-pro
        if name == "mimo" and "model_name" not in config:
            config["model_name"] = "mimo-v2.5-pro"
        if name == "mimo" and "base_url" not in config:
            config["base_url"] = "https://api.xiaomimimo.com/v1"

        return config
