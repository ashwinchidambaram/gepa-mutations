"""Default experiment config matching GEPA paper parameters for Qwen3-8B."""

# Model configuration (via OpenRouter)
MODEL = "openrouter/qwen/qwen3-8b"
REFLECTION_LM = MODEL  # Paper uses same model for reflection

# Inference parameters (Table 7 / Appendix B)
TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 20
MAX_CONTEXT = 16384

# GEPA hyperparameters (Section 4 / Appendix B)
MINIBATCH_SIZE = 3
MODULE_SELECTION = "round_robin"
MERGE_MAX_INVOCATIONS = 5

# Data splits
TRAIN_SIZE = 150
VAL_SIZE = 300
TEST_SIZE = 300

# Benchmarks to run
BENCHMARKS = [
    "hotpotqa",
    "ifbench",
    "aime",
    "livebench",
    "hover",
    "pupa",
]
