"""Experiment configuration matching GEPA paper parameters."""

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

    # Test evaluation
    test_eval_workers: int = 10


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
