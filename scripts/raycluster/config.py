"""Raycluster-specific configuration.

All cluster infrastructure settings live here. Imported by runner scripts.
"""

# Cluster infrastructure
CLUSTER_VM = "10.0.50.65"
CLUSTER_VM_HOST = "gho-vm-2.lab.supportvectors.ai"
CLUSTER_USER = "achidamb"
SSH_KEY = "~/.ssh/id_ed25519"

# Inference API
INFERENCE_HOST = "10.0.10.66"
INFERENCE_PORT = 8123
INFERENCE_BASE_URL = f"http://{INFERENCE_HOST}:{INFERENCE_PORT}/v1"
MODEL_NAME = "openai/gpt-oss-120b"  # Alias for Qwen/Qwen3.5-27B
MODEL_TAG = "qwen3.5-27b"  # For results directory naming
MODEL_FULL_NAME = "Qwen/Qwen3.5-27B"

# Model behavior
# Reasoning model with thinking disabled via extra_body.chat_template_kwargs.
# With thinking disabled, responses come cleanly in the content field.
DISABLE_THINKING = True  # Passes enable_thinking=false to suppress CoT in output
MAX_TOKENS_QA = 512  # HotpotQA, HoVer, PUPA, IFBench (matches GEPA paper)
MAX_TOKENS_MATH = 4096  # AIME, LiveBench — needs to show work (matches GEPA paper)
MAX_TOKENS_REFLECT = 4096  # Reflection LM for prompt generation (increased from 2048 — ISO prompts are longer)

# Benchmark → max_tokens mapping
BENCHMARK_MAX_TOKENS = {
    "hotpotqa": MAX_TOKENS_QA,
    "hover": MAX_TOKENS_QA,
    "pupa": MAX_TOKENS_QA,
    "ifbench": MAX_TOKENS_QA,
    "livebench": MAX_TOKENS_MATH,
    "aime": MAX_TOKENS_MATH,
}

# Experiment settings
INFRA_TAG = "raycluster"
BENCHMARKS = ["hotpotqa", "hover", "pupa", "ifbench", "livebench"]
SEEDS = [42, 123, 456, 789, 1024]

# Parallelism — vLLM handles concurrent requests via continuous batching.
# KV cache usage is ~1.3% at 6 concurrent requests, so 16 workers is safe.
# Math benchmarks (4096 tokens) need fewer workers to avoid timeouts under load.
PARALLEL_WORKERS = 16
BENCHMARK_PARALLEL_WORKERS = {
    "hotpotqa": 16,
    "hover": 16,
    "pupa": 16,
    "ifbench": 16,
    "livebench": 8,  # 4096 max_tokens — increased from 4 (KV at 0.3%, safe headroom)
    "aime": 12,      # 4096 max_tokens — increased from 4 (KV at 0.3%, safe headroom)
}

# Hyperparameters (match GEPA paper)
TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 20

# Paths (on cluster)
CLUSTER_PROJECT_DIR = "~/projects/gepa-mutations"
CLUSTER_RUNS_DIR = f"{CLUSTER_PROJECT_DIR}/runs/{MODEL_TAG}"
