"""Experiment configuration matching GEPA paper parameters."""

from __future__ import annotations

import re
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
    gepa_base_url: str = ""
    model_prefix: str = "openrouter"
    api_key: str = ""

    # Reflection/proposal model (Framing B analyzer factor).
    # When set, reflection calls use this model instead of gepa_model.
    # In GEPA: the reflector. In MIPROv2: the proposer. In ISO: discovery/mutation LM.
    # Structurally analogous but mechanically distinct across methods.
    reflection_model: str = ""       # empty = use gepa_model (self analyzer)
    reflection_base_url: str = ""    # empty = use gepa_base_url
    reflection_api_key: str = ""     # empty = use api_key

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

    Uses token-boundary matching so that the "-4bit" quantization suffix in MLX model
    IDs (e.g. "Qwen3-32B-4bit") does not trigger the "4b" size check.  A size token
    is considered a match only when it is preceded by a non-alphanumeric character and
    followed by a non-alphanumeric character or end-of-string.
    """
    m = settings.gepa_model.lower()

    def _has_size(size: str) -> bool:
        """Return True if `size` appears as a standalone token in the model name."""
        return bool(re.search(rf'(?:^|[^a-z0-9]){re.escape(size)}(?:[^a-z0-9]|$)', m))

    # Gemma family — detect before generic size checks
    if "gemma" in m:
        for size in ["27b", "12b", "4b", "1b"]:
            if _has_size(size):
                return f"gemma3-{size}"
        return "gemma3"

    # Llama family
    if "llama" in m:
        for size in ["70b", "8b", "3b", "1b"]:
            if _has_size(size):
                return f"llama3.2-{size}"
        return "llama"

    # Qwen3 — check largest sizes first to avoid "14b" matching "4b"
    for size, tag in [
        ("32b", "qwen3-32b"),
        ("27b", "qwen3-27b-awq"),
        ("14b", "qwen3-14b"),
        ("8b", "qwen3-8b"),
        ("4b", "qwen3-4b"),
        ("1.7b", "qwen3-1.7b"),
        ("0.6b", "qwen3-0.6b"),
        ("1b", "qwen3-1b"),
    ]:
        if _has_size(size):
            return tag

    # Fallback: sanitize the model name
    return m.replace("/", "-").replace(":", "-").replace(" ", "-")


def api_base_kwargs(settings: Settings) -> dict[str, Any]:
    """Return api_base/api_key kwarg dict if a custom endpoint is configured."""
    if settings.gepa_base_url:
        kw: dict[str, Any] = {"api_base": settings.gepa_base_url}
        if settings.api_key:
            kw["api_key"] = settings.api_key
        return kw
    return {}


# extra_body to disable thinking on Qwen3-series models (thinking is on by default)
THINKING_DISABLED_EXTRA_BODY: dict[str, Any] = {
    "chat_template_kwargs": {"enable_thinking": False},
    "include_reasoning": False,
}
