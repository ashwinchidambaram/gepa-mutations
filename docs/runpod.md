# RunPod Sweep Plan

## Goal

Run a clean, from-scratch 5-seed sweep for 3 model sizes (1.7B, 4B, 8B) on RunPod GPU cloud. This produces a consistent dataset on identical hardware for cross-model comparison.

**Total tasks:** 93 per model (6 methods x 3 benchmarks x 5 seeds + 3 baselines) = **279 tasks across 3 models**.

---

## Method Selection

Based on analysis of 44 complete 5-seed results from the SLURM cluster sweep, we trimmed from 12 methods to 6. This cuts cost ~50% while keeping all methods that showed meaningful signal.

### Keep (6 methods)

| Method | Type | Rationale |
|--------|------|-----------|
| `gepa` | Baseline | Paper-faithful control (ICLR 2026 Oral). Required for all comparisons. |
| `best_of_k_K3` | Proposer-Replacement | Never below baseline in any cell. Top performer on multiple benchmarks. The reliable choice. |
| `contrastive_reflection` | Proposer-Replacement | Zero extra LLM cost, always above baseline. Good "free lunch" contrast vs Best-of-K's expensive approach. |
| `synaptic_pruning` | Standalone Search | Best single result in the sweep (27B IFBench +8.7pp). 15-100x fewer rollouts than GEPA. The efficiency story. |
| `iso` | Standalone Search | Dominates PUPA across all model scales. Inter-round mutation mechanism is unique. |
| `tournament` | Standalone Search | Strong on PUPA and 27B HotpotQA. Contrast with Slime Mold — same population-based approach but without refinement. |

### Dropped (6 methods)

| Method | Reason |
|--------|--------|
| `failure_stratified_k_K3` | Zero 5-seed complete cells. Just Best-of-K with partitioned failures — no evidence it adds value. |
| `active_minibatch` | Only 2 complete cells. Disagreement signal was empirically weak (0.003-0.031 range). |
| `contrastive_synthesis` | Only 2 complete cells. Same scores as Contrastive Reflection — the extra synthesis LLM call doesn't help. Redundant. |
| `ecological_succession` | Only 1 complete cell, weakest mutation there. Curriculum learning didn't translate. Seed-sensitive. |
| `modular` | Only 2 complete cells, marginal gains. Composition step actively hurt performance in 5/6 runs. |
| `ant_colony` | Only 2 complete cells, 1 below baseline. Component decomposition poorly matched to these benchmarks. |

---

## GPU Requirements

| Model | Weights | Min VRAM | Recommended GPU | Quantization |
|-------|---------|----------|-----------------|-------------|
| Qwen3-1.7B | ~3.4 GB | 8 GB | Any 24GB GPU | bf16 (native) |
| Qwen3-4B | ~8 GB | 16 GB | Any 24GB GPU | bf16 (native) |
| Qwen3-8B | ~16 GB | 24 GB | RTX 4090 / A5000 | bf16 (native) |
| Qwen3-14B | ~28 GB | 48 GB | RTX A6000 / L40S | fp8 required |
| Qwen3-27B-AWQ | ~14 GB | 32 GB | RTX 5090 | AWQ 4-bit |

**14B and 27B do NOT fit on 24GB GPUs.** Only run 1.7B, 4B, and 8B on RTX 4090 / A5000 / 3090.

---

## Pod Configuration

### Option A: Single Multi-GPU Pod (Recommended)

One pod with 3 GPUs. Cheapest, simplest, shared filesystem.

- **Template:** RunPod PyTorch 2.8
- **GPUs:** 3x RTX 4090 (or 3x A5000, 3x 3090 — any 24GB GPU works)
- **Volume:** 50 GB at `/workspace` (persistent)
- **Expose ports:** 8125, 8126, 8127

### Option B: Separate Pods

3 individual pods, 1 GPU each. More flexible (can use different GPU types) but higher disk cost and more complex networking.

---

## Cost Estimates

### RTX 4090 pricing (~$0.59/GPU/hr)

| Phase | Models | Est. Wall Time | Cost |
|-------|--------|---------------|------|
| Phase 1 | 1.7B + 8B (2 workers) | ~20h | ~$35 |
| Phase 2 | Swap 1.7B pod → 4B | ~15h | ~$9* |
| **Total** | | ~35h wall | **~$45** |

*Phase 2 only uses 1 GPU while 8B continues on the other 2.

### RTX A5000 pricing (~$0.16/GPU/hr) — if available

| Phase | Models | Est. Wall Time | Cost |
|-------|--------|---------------|------|
| Phase 1 | 1.7B + 8B (2 workers) | ~20h | ~$10 |
| Phase 2 | Swap → 4B | ~15h | ~$2 |
| **Total** | | ~35h wall | **~$12** |

### RTX 3090 pricing (~$0.22/GPU/hr) — if available

Similar to A5000, total ~$17.

**Check availability at deploy time — GPU stock fluctuates. Cheapest 24GB GPU available is the best choice.**

---

## Setup Script

Paste this into the pod terminal after SSH-ing in:

```bash
#!/bin/bash
# === RunPod Setup Script for GEPA Mutations Sweep ===

# 1. Install vLLM
pip install vllm --quiet

# 2. Clone repo
cd /workspace
git clone https://github.com/ashwinchidambaram/gepa-mutations.git
cd gepa-mutations

# 3. Install project dependencies
# If using uv:
# uv sync
# If using pip:
pip install -e . --quiet

# 4. Download models (cached on persistent volume)
huggingface-cli download Qwen/Qwen3-1.7B
huggingface-cli download Qwen/Qwen3-8B
# huggingface-cli download Qwen/Qwen3-4B  # uncomment when ready for phase 2

echo "Setup complete!"
```

---

## Running the Sweep

### Step 1: Start vLLM Servers (one per GPU)

Open 3 terminal sessions (or use tmux/screen):

```bash
# Terminal 1: 1.7B on GPU 0, port 8127
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-1.7B \
  --dtype auto \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --host 0.0.0.0 \
  --port 8127

# Terminal 2: 8B on GPU 1, port 8125
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-8B \
  --dtype auto \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --host 0.0.0.0 \
  --port 8125

# Terminal 3: 8B on GPU 2, port 8126 (second 8B worker)
CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-8B \
  --dtype auto \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --host 0.0.0.0 \
  --port 8126
```

### Step 2: Verify Endpoints

```bash
curl -s http://localhost:8127/v1/models | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])"
curl -s http://localhost:8125/v1/models | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])"
curl -s http://localhost:8126/v1/models | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])"
```

### Step 3: Clear Old Results (Fresh Start)

```bash
cd /workspace/gepa-mutations

# Remove old 1.7B and 8B results to start clean
rm -rf runs/qwen3-1.7b runs/qwen3-8b

# Verify clean
find runs/ -name result.json | wc -l  # should be 0 for these models
```

### Step 4: Run Baseline Evaluations First

```bash
cd /workspace/gepa-mutations

# Run baselines (if baseline runner exists)
# Otherwise, orchestrator will handle this

# 1.7B baselines
GEPA_MODEL="Qwen/Qwen3-1.7B" GEPA_BASE_URL="http://localhost:8127/v1" \
  python scripts/run_baseline.py --benchmark hotpotqa pupa ifbench

# 8B baselines
GEPA_MODEL="Qwen/Qwen3-8B" GEPA_BASE_URL="http://localhost:8125/v1" \
  python scripts/run_baseline.py --benchmark hotpotqa pupa ifbench
```

### Step 5: Launch Orchestrators

```bash
cd /workspace/gepa-mutations

# Methods to run (trimmed from 12 to 6 based on SLURM sweep analysis)
METHODS="gepa best_of_k_K3 contrastive_reflection synaptic_pruning iso tournament"

# 1.7B orchestrator (all benchmarks)
GEPA_MODEL="Qwen/Qwen3-1.7B" GEPA_BASE_URL="http://localhost:8127/v1" \
  python scripts/run_all_local.py --workers 8 --benchmark hotpotqa pupa ifbench \
  --method $METHODS \
  > logs/orchestrator_1b_runpod.log 2>&1 &
echo "1.7B PID: $!"

# 8B orchestrator — GPU 1 (hotpotqa + pupa)
GEPA_MODEL="Qwen/Qwen3-8B" GEPA_BASE_URL="http://localhost:8125/v1" \
  python scripts/run_all_local.py --workers 6 --benchmark hotpotqa pupa \
  --method $METHODS \
  > logs/orchestrator_8b_runpod.log 2>&1 &
echo "8B-main PID: $!"

# 8B orchestrator — GPU 2 (ifbench)
GEPA_MODEL="Qwen/Qwen3-8B" GEPA_BASE_URL="http://localhost:8126/v1" \
  python scripts/run_all_local.py --workers 6 --benchmark ifbench \
  --method $METHODS \
  > logs/orchestrator_8b_ifbench_runpod.log 2>&1 &
echo "8B-ifbench PID: $!"
```

### Step 6: Monitor Progress

```bash
# Result counts
python3 -c "
import glob,collections
c=collections.Counter()
for f in glob.glob('runs/qwen3-*/*/*/*/result.json'):
    c[f.split('/')[1]] += 1
[print(k,v) for k,v in sorted(c.items())]
print('Total:',sum(c.values()))
"

# Check orchestrator logs
tail -5 logs/orchestrator_*_runpod.log

# vLLM health
for port in 8125 8126 8127; do
  echo "=== :$port ==="
  curl -s http://localhost:$port/metrics | grep -E "kv_cache_usage_perc|num_requests_running"
done
```

---

## Phase 2: Swap 1.7B → 4B

Once 1.7B completes all 183 tasks:

```bash
# Kill 1.7B vLLM server (find its PID)
kill <1.7B_vLLM_PID>

# Download 4B model
huggingface-cli download Qwen/Qwen3-4B

# Start 4B on the same GPU
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-4B \
  --dtype auto \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90 \
  --host 0.0.0.0 \
  --port 8127

# Launch 4B orchestrator (same 6 methods)
METHODS="gepa best_of_k_K3 contrastive_reflection synaptic_pruning iso tournament"
GEPA_MODEL="Qwen/Qwen3-4B" GEPA_BASE_URL="http://localhost:8127/v1" \
  python scripts/run_all_local.py --workers 8 --benchmark hotpotqa pupa ifbench \
  --method $METHODS \
  > logs/orchestrator_4b_runpod.log 2>&1 &
```

---

## Validation

After sweep completes, validate all results:

```bash
python3 -c "
import glob,json
bad=[]
for f in glob.glob('runs/qwen3-*/*/*/*/result.json'):
    try:
        d=json.load(open(f))
        ts=d.get('test_score','MISSING')
        bench=f.split('/')[2]
        m=json.load(open(f.replace('result.json','metrics.json')))
        tt=m.get('total_tokens',0)
        if ts=='MISSING' or ts is None: bad.append(f'NONE: {f}')
        elif ts==0.0: bad.append(f'ZERO_SCORE: {f} tokens={tt}')
        elif tt==0: bad.append(f'ZERO_TOKENS: {f}')
    except Exception as e: bad.append(f'ERR: {f}: {e}')
print('All CLEAN' if not bad else '\n'.join(bad))
print(f'Scanned: {len(list(glob.glob(\"runs/qwen3-*/*/*/*/result.json\")))} files')
"
```

---

## After Completion

```bash
# Push results to git
cd /workspace/gepa-mutations
git add runs/ logs/
git commit -m "data: clean RunPod sweep — 1.7B/4B/8B on RTX 4090"
git push

# STOP THE POD to avoid charges!
# Do this from RunPod dashboard or:
# runpodctl stop pod <pod-id>
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| vLLM OOM on startup | Lower `--gpu-memory-utilization` to 0.85 |
| Model download slow | Models cache on `/workspace` volume — only slow first time |
| Orchestrator dies | It's idempotent — just restart, it skips completed tasks |
| All results score 0.0 | vLLM likely crashed — check vLLM terminal for errors |
| Port already in use | Find and kill: `lsof -i :8125` then `kill <PID>` |
| Pod stopped unexpectedly | Volume persists — restart pod, re-run vLLM + orchestrator |
