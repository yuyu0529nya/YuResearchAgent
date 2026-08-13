"""Resolve secret-free model identities for reproducible evaluations."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit, urlunsplit

from src.models.model_router import ModelRouter

MODULE_NAMES = (
    "solver",
    "planner",
    "summarizer",
    "judge",
    "red_agent",
    "blue_agent",
    "compressor",
)


def configure_single_backend(config: dict[str, Any], backend: str) -> None:
    config.setdefault("model", {})["backend"] = backend
    config["model"]["backend_mapping"] = {module: backend for module in MODULE_NAMES}


def requested_sampling(config: dict[str, Any], backend: str, module: str) -> dict[str, Any]:
    sampling = config.get("model", {}).get("backend_sampling", {})
    resolved = dict(sampling.get(backend, {}))
    resolved.update(sampling.get("modules", {}).get(module, {}))
    return {key: resolved[key] for key in ("temperature", "top_p", "max_tokens") if key in resolved}


def _safe_endpoint(policy: Any) -> str:
    raw = str(getattr(getattr(policy, "client", None), "base_url", ""))
    try:
        parsed = urlsplit(raw)
    except ValueError:
        return ""
    if not parsed.scheme or not parsed.netloc:
        return ""
    host = parsed.hostname or ""
    if parsed.port:
        host = f"{host}:{parsed.port}"
    return urlunsplit((parsed.scheme, host, parsed.path, "", "")).rstrip("/")


def policy_identity(policy: Any) -> dict[str, Any]:
    """Return output-affecting policy settings while excluding credentials."""
    return {
        "model": str(policy.model_name),
        "endpoint": _safe_endpoint(policy),
        "temperature": policy.temperature,
        "top_p": policy.top_p,
        "max_tokens": policy.max_tokens,
        "extra_body": dict(getattr(policy, "extra_body", None) or {}),
    }


def create_policy(config: dict[str, Any], backend: str, module: str) -> Any:
    return ModelRouter.create_backend(
        backend,
        use_cache=False,
        **requested_sampling(config, backend, module),
    )


def effective_module_policies(config: dict[str, Any], backend: str) -> dict[str, dict[str, Any]]:
    return {module: policy_identity(create_policy(config, backend, module)) for module in ("default", *MODULE_NAMES)}


__all__ = [
    "MODULE_NAMES",
    "configure_single_backend",
    "create_policy",
    "effective_module_policies",
    "policy_identity",
    "requested_sampling",
]
