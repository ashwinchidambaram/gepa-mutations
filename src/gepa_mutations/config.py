"""Experiment configuration matching GEPA paper parameters."""

from __future__ import annotations

from typing import Any

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Environment-based settings for the experiment runner."""

    model_config = {"env_prefix": "", "env_file": ".env", "extra": "ignore"}

    # OpenRouter
    openrouter_api_key: str = ""

    # HuggingFace
    hf_token: str = ""

    # Telegram
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # AWS
    s3_bucket: str = "gepa-mutations-results"

    # Model settings (paper defaults for Qwen3-8B)
    gepa_model: str = "qwen/qwen3-8b"
    gepa_temperature: float = 0.6
    gepa_top_p: float = 0.95
    gepa_top_k: int = 20
    gepa_max_context: int = 16384

    # Model endpoint (empty = provider default, e.g. OpenRouter)
    api_base_url: str = ""
    model_prefix: str = "openrouter"
    api_key: str = ""

    # Test evaluation
    test_eval_workers: int = 10

    # LM call settings (tuned for local non-thinking inference, 0.5ms LAN)
    lm_timeout: int = 60
    max_tokens_qa: int = 512        # hotpotqa, hover, pupa, ifbench — short answers
    max_tokens_math: int = 4096     # aime, livebench — needs to show work; 1024 truncates AIME reasoning
    max_tokens_reflection: int = 2048  # reflection LM generates new system prompts


# Paper baseline scores for comparison (Table 1: Qwen3-8B test set)
PAPER_BASELINES = {
    "qwen3-8b": {
        "baseline": {
            "hotpotqa": 42.33, "ifbench": 36.90, "hover": 35.33,
            "pupa": 80.82, "aime": 27.33, "livebench": 48.70,
            "aggregate": 45.23,
        },
        "grpo": {
            "hotpotqa": 43.33, "ifbench": 35.88, "hover": 38.67,
            "pupa": 86.66, "aime": 38.00, "livebench": 51.26,
            "aggregate": 48.91,
        },
        "miprov2": {
            "hotpotqa": 55.33, "ifbench": 36.22, "hover": 47.33,
            "pupa": 81.55, "aime": 20.00, "livebench": 46.60,
            "aggregate": 47.84,
        },
        "gepa": {
            "hotpotqa": 62.33, "ifbench": 38.61, "hover": 52.33,
            "pupa": 91.85, "aime": 32.00, "livebench": 51.95,
            "aggregate": 54.85,
        },
        "gepa_merge": {
            "hotpotqa": 64.33, "ifbench": 28.23, "hover": 51.67,
            "pupa": 86.26, "aime": 32.00, "livebench": 51.95,
            "aggregate": 52.40,
        },
    },
}

# Paper rollout budgets (Table 1)
PAPER_ROLLOUTS = {
    "gepa": {
        "hotpotqa": 6871, "ifbench": 3593, "aime": 7051,
        "hover": 2426, "livebench": 1839, "pupa": 3936,
    },
    "grpo": {
        "hotpotqa": 24000, "ifbench": 24000, "aime": 24000,
        "hover": 24000, "livebench": 24000, "pupa": 24000,
    },
}

# Paper hyperparameters
PAPER_HYPERPARAMS = {
    "minibatch_size": 3,
    "module_selection": "round_robin",
    "merge_max_invocations": 5,
    "qwen3_8b": {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "max_context": 16384,
    },
}


# ---------------------------------------------------------------------------
# Model helpers (here to avoid circular imports between base.py / experiment.py)
# ---------------------------------------------------------------------------


def model_id(settings: Settings) -> str:
    """Build the full LiteLLM model ID from settings."""
    return f"{settings.model_prefix}/{settings.gepa_model}"


def model_tag(settings: Settings) -> str:
    """Derive a filesystem-safe model tag from the GEPA_MODEL env var.

    Used to namespace result directories: runs/{model_tag}/{benchmark}/{method}/{seed}/
    """
    m = settings.gepa_model.lower()
    if "27b" in m:
        return "qwen3-27b-awq"
    if "8b" in m:
        return "qwen3-8b"
    if "4b" in m:
        return "qwen3-4b"
    if "1.7b" in m or "1b" in m:
        return "qwen3-1.7b"
    # Fallback: sanitize the model name
    return m.replace("/", "-").replace(":", "-").replace(" ", "-")


def api_base_kwargs(settings: Settings) -> dict[str, Any]:
    """Return api_base/api_key kwarg dict if a custom endpoint is configured."""
    if settings.api_base_url:
        kw: dict[str, Any] = {"api_base": settings.api_base_url}
        if settings.api_key:
            kw["api_key"] = settings.api_key
        return kw
    return {}


# extra_body to disable thinking on Qwen3-series models (thinking is on by default)
THINKING_DISABLED_EXTRA_BODY: dict[str, Any] = {
    "chat_template_kwargs": {"enable_thinking": False},
    "include_reasoning": False,
}
