# Mac Setup (sweep/mac branch)

## Hardware

- MacBook Pro M5 Max
- 40-core GPU (unified memory architecture)
- 48GB unified memory — shared between CPU and GPU
- No SLURM, no NFS, no HF_HOME needed

---

## MLX-LM

MLX-LM serves an OpenAI-compatible API on Apple Silicon. The orchestrator
(`scripts/run_all_local.py`) uses `GEPA_BASE_URL` to point to it — drop-in replacement
for vLLM with no code changes.

```bash
pip install mlx-lm

# Serve command pattern
python -m mlx_lm.server --model <model-id> --port <port> --host 127.0.0.1
```

No `--enforce-eager`, `--dtype`, or `--gpu-memory-utilization` — those are vLLM-only flags.

---

## Models and Ports

| Model | Family | MLX model ID | Port | ~Memory |
|-------|--------|-------------|------|---------|
| Qwen3-0.6B | Qwen3 | `mlx-community/Qwen3-0.6B-4bit` | 8132 | 0.4 GB |
| Qwen3-1.7B | Qwen3 | `mlx-community/Qwen3-1.7B-4bit` | 8125 | 1 GB |
| Qwen3-4B | Qwen3 | `mlx-community/Qwen3-4B-4bit` | 8126 | 3 GB |
| Qwen3-32B | Qwen3 | `mlx-community/Qwen3-32B-4bit` | 8131 | 20 GB |
| Gemma 3 1B | Gemma 3 | `mlx-community/gemma-3-1b-it-4bit` | 8133 | 0.6 GB |
| Gemma 3 4B | Gemma 3 | `mlx-community/gemma-3-4b-it-4bit` | 8134 | 2.5 GB |
| Gemma 3 12B | Gemma 3 | `mlx-community/gemma-3-12b-it-4bit` | 8135 | 7 GB |
| Gemma 3 27B | Gemma 3 | `mlx-community/gemma-3-27b-it-4bit` | 8136 | 15 GB |
| Llama 3.2 1B | Llama | `mlx-community/Llama-3.2-1B-Instruct-4bit` | 8137 | 0.6 GB |
| Llama 3.2 3B | Llama | `mlx-community/Llama-3.2-3B-Instruct-4bit` | 8138 | 2 GB |

---

## Memory Groupings

48GB doesn't hold all 10 models simultaneously (~52GB total). Run in two groups:

**Group A — small models (~14GB): run all together**
```
Qwen3-0.6B + Qwen3-1.7B + Qwen3-4B
Gemma3-1B + Gemma3-4B
Llama3-1B + Llama3-3B
```

**Group B — large models (~42GB): run all together**
```
Qwen3-32B (~20GB) + Gemma3-12B (~7GB) + Gemma3-27B (~15GB)
```

You can run Group A and Group B sequentially, or interleave them if you're willing to kill
and restart servers. The orchestrators are idempotent — they pick up where they left off.

---

## Orchestration Execution Order

Each model gets its own independent orchestrator process. The orchestrator sorts experiments
**fastest-first within each benchmark** using `EXPERIMENT_DURATION_MINS` in `run_all_local.py`.

**Benchmark order (all models):** `ifbench → pupa → livebench → hotpotqa → hover`

**Important for new model families (Gemma 3, Llama):** `EXPERIMENT_DURATION_MINS` was
measured on Qwen3-27B runs. For Gemma and Llama models, there's no timing data yet — those
experiments fall back to `METHOD_PRIORITY` order:

```
gepa → best_of_k_K3 → synaptic_pruning → iso → tournament → ant_colony →
active_minibatch → ecological_succession → modular → contrastive_synthesis →
contrastive_reflection → failure_stratified_k_K3
```

This is a reasonable fallback for first runs. Once experiments complete, you can add actual
timing observations to `EXPERIMENT_DURATION_MINS` for subsequent runs.

**Recommended launch order for the Mac:** Start small models first (Group A) since they
complete fastest and give early signal on whether cross-architecture results look sane.
Then run Group B (large models) in parallel while Group A is finishing.

---

## Worker Count Guidelines

| Model | Recommended workers | Notes |
|-------|--------------------|----|
| 0.6B / 1.7B | 6–8 | Tiny models, saturate quickly |
| 4B / Gemma3-4B / Llama3-3B | 5–6 | Good throughput |
| Gemma3-12B | 3–4 | Mid-size, moderate concurrency |
| 32B / Gemma3-27B | 2–3 | Large models, keep workers low |

---

## Model Download

Pre-download all models before starting:

```bash
python -c "
from mlx_lm import load
models = [
    'mlx-community/Qwen3-0.6B-4bit',
    'mlx-community/Qwen3-1.7B-4bit',
    'mlx-community/Qwen3-4B-4bit',
    'mlx-community/Qwen3-32B-4bit',
    'mlx-community/gemma-3-1b-it-4bit',
    'mlx-community/gemma-3-4b-it-4bit',
    'mlx-community/gemma-3-12b-it-4bit',
    'mlx-community/gemma-3-27b-it-4bit',
    'mlx-community/Llama-3.2-1B-Instruct-4bit',
    'mlx-community/Llama-3.2-3B-Instruct-4bit',
]
for m in models:
    print(f'Downloading {m}...')
    load(m)
    print('  done')
"
```

Qwen3-32B-4bit is ~20GB — allow time on first run.

---

## Monitoring Without Crons

```bash
# Quick progress check across all models
for tag in qwen3-0.6b qwen3-1.7b qwen3-4b qwen3-32b \
           gemma3-1b gemma3-4b gemma3-12b gemma3-27b \
           llama3-1b llama3-3b; do
  count=$(find runs/$tag -name result.json 2>/dev/null | wc -l | tr -d ' ')
  echo "$tag: $count / 300"
done

# Send Telegram update manually
.venv/bin/python scripts/monitor_multi_model.py --mode 30min

# Watch a specific log
tail -f logs/orchestrator_gemma3_27b.log
```
