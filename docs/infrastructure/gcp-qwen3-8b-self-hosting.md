# Self-Hosting Qwen3-8B on GCP: Cost Analysis

**Date:** 2026-03-22
**Model:** Qwen/Qwen3-8B (8.2B dense parameters)
**Budget Ceiling:** $100/month hard limit
**Current approach:** OpenRouter API at $0.05/$0.40 per M tokens (in/out)
**Current cost per experiment:** $0.50-$5.00
**Experiment runtime:** 12-24 hours each
**Experiment volume:** 2-4 experiments/month

---

## Executive Summary

**Recommendation: Self-hosting Qwen3-8B on GCP is viable and likely cheaper than OpenRouter at your experiment volume -- but the margin is thin, and the operational overhead is the real cost.**

Qwen3-8B is a small model. It fits on a single T4 (16GB VRAM) at INT8, or comfortably on a single L4 (24GB) at FP16. The cheapest GCP option -- an n1-standard-2 with 1x T4 on spot -- costs ~$0.14-$0.17/hr total, putting a 24-hour experiment at $3.36-$4.08. The g2-standard-4 (1x L4) at $0.28/hr spot costs $3.38-$6.76 per experiment. Both are competitive with OpenRouter's $0.50-$5.00 range, but only the T4 option consistently wins.

The key advantage is not cost -- it is **scientific comparability**. Self-hosting the exact same Qwen3-8B model means your results are directly comparable to the paper baselines. No API provider quirks, no quantization differences, no rate limits. Same model, same weights, controlled inference parameters.

---

## Qwen3-8B VRAM Requirements

| Precision | VRAM (Weights Only) | VRAM (With KV Cache + Overhead) | Notes |
|-----------|--------------------|---------------------------------|-------|
| FP16/BF16 | ~16 GB | ~18-20 GB | Fits on L4 (24GB) with 4-6GB headroom |
| INT8 | ~8 GB | ~10-12 GB | Fits on T4 (16GB) with 4-6GB headroom |
| INT4/AWQ | ~4-5 GB | ~6-8 GB | Fits on T4 with massive headroom |

**Model download size:** ~16 GB (FP16/BF16 weights from HuggingFace)

This is a fundamentally different sizing problem than gpt-oss-20b (41GB) or Qwen2.5-VL-72B (144GB). A single entry-level GPU handles it without quantization compromises.

---

## GCP Instance Recommendations for Qwen3-8B

### Option 1: n1-standard-2 + 1x T4 (Budget Champion)

| Component | On-Demand ($/hr) | Spot ($/hr) |
|-----------|-----------------|-------------|
| n1-standard-2 (2 vCPU, 7.5GB RAM) | $0.095 | ~$0.028 |
| 1x NVIDIA T4 (16GB VRAM) | $0.35 | ~$0.11-$0.14 |
| **Total** | **~$0.45** | **~$0.14-$0.17** |

- **VRAM:** 16 GB. Runs Qwen3-8B at INT8 (~10GB with overhead). Tight but workable.
- **Quantization required:** INT8 or INT4. FP16 will not fit with KV cache overhead.
- **GPU compute:** 65 TFLOPS FP16, 130 TOPS INT8. Memory bandwidth: 320 GB/s.
- **Throughput estimate:** ~15-40 tok/s (single request), ~50-100 tok/s (batched). The T4 is memory bandwidth-limited for autoregressive decoding.
- **Availability:** T4 spot is widely available in us-central1. Eviction rate is low because T4 demand has dropped with L4/H100 adoption.

### Option 2: g2-standard-4 + 1x L4 (Performance Sweet Spot)

| Component | On-Demand ($/hr) | Spot ($/hr) |
|-----------|-----------------|-------------|
| g2-standard-4 (4 vCPU, 16GB RAM, 1x L4) | $0.71 | $0.28 |

- **VRAM:** 24 GB. Runs Qwen3-8B at FP16 (~18-20GB with overhead). No quantization needed.
- **GPU compute:** 242 TFLOPS FP16 (via FP8 Tensor Cores), 30.3 TFLOPS native FP16. Memory bandwidth: 395 GB/s.
- **Throughput estimate:** ~30-80 tok/s (single request), ~100-300 tok/s (batched). The L4 delivers ~8x better throughput than T4 for comparable models.
- **Key advantage:** No quantization means output quality is identical to the paper's configuration. FP16 inference = bit-exact reproducibility.

### Option 3: g2-standard-8 + 1x L4 (If You Need More CPU/RAM)

| Component | On-Demand ($/hr) | Spot ($/hr) |
|-----------|-----------------|-------------|
| g2-standard-8 (8 vCPU, 32GB RAM, 1x L4) | $0.85 | $0.34 |

- Same GPU as g2-standard-4. More CPU and RAM for the GEPA orchestration layer.
- Only justified if the GEPA runner itself needs >16GB RAM or >4 vCPUs for concurrent scoring, dataset loading, etc. Unlikely for this workload.

### Option 4: a2-highgpu-1g + 1x A100-40GB (Overkill)

| Component | On-Demand ($/hr) | Spot ($/hr) |
|-----------|-----------------|-------------|
| a2-highgpu-1g (12 vCPU, 85GB RAM, 1x A100-40GB) | $3.67 | $1.80 |

- **VRAM:** 40 GB. Qwen3-8B at FP16 uses <20GB. You are paying for 20GB of unused VRAM.
- **Throughput:** ~200-500+ tok/s. Massive overkill for a model this small.
- **Verdict:** No. The A100 costs 6.4x more than the L4 for a model that fits comfortably on the L4. The throughput increase does not reduce experiment time proportionally because the bottleneck shifts to scoring and pipeline overhead.

### Instance Comparison Summary

| Instance | GPU | VRAM | Spot $/hr | Qwen3-8B Fits? | Quantization | Est. tok/s | Best For |
|----------|-----|------|-----------|-----------------|--------------|------------|----------|
| n1-standard-2 + T4 | T4 | 16 GB | ~$0.14-0.17 | Yes (INT8) | Required | ~15-40 | **Lowest cost** |
| g2-standard-4 (L4) | L4 | 24 GB | $0.28 | Yes (FP16) | Not needed | ~30-80 | **Best value** |
| g2-standard-8 (L4) | L4 | 24 GB | $0.34 | Yes (FP16) | Not needed | ~30-80 | Extra CPU/RAM |
| a2-highgpu-1g (A100) | A100-40GB | 40 GB | $1.80 | Yes (FP16) | Not needed | ~200-500 | **Overkill** |

---

## Cost Per Experiment

### n1-standard-2 + T4 spot (~$0.17/hr)

| Scenario | Spot Cost | On-Demand Cost |
|----------|-----------|----------------|
| 12-hour experiment | $2.04 | $5.40 |
| 24-hour experiment | $4.08 | $10.80 |
| + Startup overhead (~15 min) | +$0.04 | +$0.11 |
| **Total (12hr)** | **$2.08** | **$5.51** |
| **Total (24hr)** | **$4.12** | **$10.91** |

### g2-standard-4 (L4) spot ($0.28/hr)

| Scenario | Spot Cost | On-Demand Cost |
|----------|-----------|----------------|
| 12-hour experiment | $3.36 | $8.52 |
| 24-hour experiment | $6.72 | $17.04 |
| + Startup overhead (~10 min) | +$0.05 | +$0.12 |
| **Total (12hr)** | **$3.41** | **$8.64** |
| **Total (24hr)** | **$6.77** | **$17.16** |

---

## Monthly Cost

### n1-standard-2 + T4 spot

| Experiments/Month | 12hr each | 24hr each |
|------------------|-----------|-----------|
| 2 | $4.16 | $8.24 |
| 3 | $6.24 | $12.36 |
| 4 | $8.32 | $16.48 |

### g2-standard-4 (L4) spot

| Experiments/Month | 12hr each | 24hr each |
|------------------|-----------|-----------|
| 2 | $6.82 | $13.54 |
| 3 | $10.23 | $20.31 |
| 4 | $13.64 | $27.08 |

**Both options are well within the $100/month budget.** The T4 spot option at 4 experiments/month of 24 hours each costs $16.48 -- 16.5% of budget. The L4 option costs $27.08 -- 27.1% of budget. This leaves $73-$84/month for storage, logging, and API fallback.

---

## Break-Even Analysis: Self-Hosting vs. OpenRouter

### At What Experiment Cost Does Self-Hosting Win?

**Using n1-standard-2 + T4 spot (24hr experiment = $4.12):**

| OpenRouter Cost/Experiment | Self-Host Cost | Self-Host Cheaper? | Savings |
|---------------------------|----------------|-------------------|---------|
| $0.50 | $4.12 | **No** (8.2x more) | -$3.62 |
| $1.00 | $4.12 | **No** (4.1x more) | -$3.12 |
| $2.50 | $4.12 | **No** (1.6x more) | -$1.62 |
| $5.00 | $4.12 | **Yes** (0.8x) | +$0.88 |

**Using n1-standard-2 + T4 spot (12hr experiment = $2.08):**

| OpenRouter Cost/Experiment | Self-Host Cost | Self-Host Cheaper? | Savings |
|---------------------------|----------------|-------------------|---------|
| $0.50 | $2.08 | **No** (4.2x more) | -$1.58 |
| $1.00 | $2.08 | **No** (2.1x more) | -$1.08 |
| $2.50 | $2.08 | **Yes** (0.8x) | +$0.42 |
| $5.00 | $2.08 | **Yes** (0.4x) | +$2.92 |

**Using g2-standard-4 (L4) spot (24hr experiment = $6.77):**

| OpenRouter Cost/Experiment | Self-Host Cost | Self-Host Cheaper? | Savings |
|---------------------------|----------------|-------------------|---------|
| $0.50 | $6.77 | **No** (13.5x more) | -$6.27 |
| $2.50 | $6.77 | **No** (2.7x more) | -$4.27 |
| $5.00 | $6.77 | **No** (1.4x more) | -$1.77 |

### Key Insight

Self-hosting only breaks even on pure cost if your OpenRouter experiments consistently cost $2.50+ (T4, 12hr) or $5.00+ (T4, 24hr). For the L4, OpenRouter needs to exceed ~$7/experiment for self-hosting to be cheaper on a per-experiment basis.

**But cost is not the only variable.** The real question is: do you need exact model control?

---

## Throughput Estimates

### Qwen3-8B on Each GPU (vLLM, Single Instance)

| GPU | Quantization | Single-Request (tok/s) | Batched (tok/s) | Source |
|-----|-------------|----------------------|-----------------|--------|
| T4 (16GB) | INT8 | ~15-40 | ~50-100 | Extrapolated from Clarifai T4 benchmarks (~3.8 tok/s for 7B at FP16, 8x with INT8 + batching) |
| L4 (24GB) | FP16 | ~30-80 | ~100-300 | Clarifai L4 benchmarks (~30 tok/s single-request for 7B), Qwen GPU benchmarks |
| L4 (24GB) | INT8 | ~50-120 | ~150-400 | L4 INT8 tensor cores provide ~2x boost over FP16 |
| A100-40GB | FP16 | ~150-400 | ~500-2,000 | Microsoft vLLM benchmarks for Llama 3.1 8B |
| A10G (AWS) | FP16 | ~40-100 | ~120-400 | Comparable to L4 with 1.5x memory bandwidth advantage |

### Throughput vs. OpenRouter

OpenRouter scales horizontally. A single self-hosted GPU cannot match OpenRouter's burst throughput because OpenRouter distributes requests across a fleet. However, GEPA experiments make **sequential** API calls (one generation at a time, or small batches). At batch size 1-4, a single L4 producing 30-80 tok/s is sufficient -- each API call in GEPA generates a candidate prompt (hundreds of tokens), not millions.

**For GEPA's access pattern (sequential, small-batch), a single L4 is fast enough.** The T4 is slower but still adequate -- GEPA experiments are not bottlenecked on raw token throughput when running for 12-24 hours.

---

## Startup Time Estimates

| Instance | Steps | Est. Total Time |
|----------|-------|----------------|
| n1-standard-2 + T4 (fresh) | Boot (2min) + driver install (3-5min) + download 16GB (5-8min) + vLLM load (2-3min) | **12-18 min** |
| n1-standard-2 + T4 (cached PD) | Boot (2min) + vLLM load (2-3min) | **4-5 min** |
| g2-standard-4 / L4 (fresh) | Boot (2min) + download 16GB (3-5min) + vLLM load (2-3min) | **7-10 min** |
| g2-standard-4 / L4 (cached PD) | Boot (2min) + vLLM load (2-3min) | **4-5 min** |
| g2-standard-4 / L4 (DL VM image) | Boot (2min) + download 16GB (3-5min) + vLLM load (2-3min) | **7-10 min** |

**Notes:**
- The g2 instances use GCP's Deep Learning VM images which have NVIDIA drivers pre-installed. The n1+T4 option may need manual driver installation (add 3-5min) unless you use the DL VM image.
- Model download at 16GB takes 3-8 minutes depending on network throughput to HuggingFace CDN from GCE.
- A pre-cached persistent disk snapshot with the model pre-downloaded eliminates the download step. A 50GB Balanced PD costs ~$5/month. Worth it if you run 2+ experiments/month.
- vLLM cold start (model loading into GPU memory) takes 2-3 minutes for an 8B model. Much faster than the 5-10 minutes for 20B+ models.

**Total startup overhead: 5-18 minutes.** This is negligible against 12-24 hour experiment runtimes.

---

## GCP vs. AWS Comparison for Qwen3-8B

### Best Instance on Each Cloud

| Metric | GCP n1-std-2 + T4 (spot) | GCP g2-std-4 / L4 (spot) | AWS g6.xlarge / L4 (spot) | AWS g5.xlarge / A10G (spot) |
|--------|--------------------------|--------------------------|---------------------------|----------------------------|
| GPU | T4 16GB | L4 24GB | L4 24GB | A10G 24GB |
| GPU TFLOPS (FP16) | 65 | 30.3 (242 w/ FP8 TC) | 30.3 (242 w/ FP8 TC) | 31.2 |
| Memory BW | 320 GB/s | 395 GB/s | 395 GB/s | 600 GB/s |
| vCPUs / RAM | 2 / 7.5 GB | 4 / 16 GB | 4 / 16 GB | 4 / 16 GB |
| Spot $/hr | ~$0.14-0.17 | $0.28 | ~$0.17 | ~$0.25 |
| Spot 12hr cost | $1.68-2.04 | $3.36 | ~$2.04 | ~$3.00 |
| Spot 24hr cost | $3.36-4.08 | $6.72 | ~$4.08 | ~$6.00 |
| Qwen3-8B fits (FP16)? | No (INT8 required) | Yes | Yes | Yes |
| Est. throughput (tok/s) | ~15-40 | ~30-80 | ~30-80 | ~40-100 |

### Head-to-Head

| Factor | GCP | AWS | Winner |
|--------|-----|-----|--------|
| Cheapest spot option | ~$0.14-0.17/hr (n1+T4) | ~$0.17/hr (g6.xlarge L4) | **GCP** (T4 is cheapest, but requires INT8) |
| Best price/performance | $0.28/hr (g2-std-4 L4) | $0.17/hr (g6.xlarge L4) | **AWS** (same GPU, 39% cheaper) |
| FP16 without quantization | $0.28/hr (L4) | $0.17/hr (L4) | **AWS** |
| Higher throughput | $0.28/hr (L4, ~30-80 tok/s) | $0.25/hr (A10G, ~40-100 tok/s) | **AWS** (A10G has 1.5x memory BW) |
| Spot availability (L4) | Good (us-central1) | Good (us-east-1) | Tie |
| Deep Learning VM images | Yes (deeplearning-platform-release) | Yes (Deep Learning AMI) | Tie |
| T4 option available | Yes (n1 + T4 attachment) | Yes (g4dn.xlarge) | Tie |

### Verdict: AWS is cheaper for L4-class instances.

The AWS g6.xlarge (1x L4, $0.17/hr spot) undercuts the GCP g2-standard-4 (1x L4, $0.28/hr spot) by 39%. Same GPU, same 24GB VRAM, same performance. GCP's only cost advantage is the T4 option at $0.14-0.17/hr, but that requires INT8 quantization which may introduce minor output differences vs. the paper's presumably FP16 configuration.

If you are already on AWS (which this project currently uses), there is zero reason to move to GCP for this workload. The g6.xlarge is cheaper and equally capable.

---

## The Real Argument for Self-Hosting: Scientific Comparability

This is not a cost optimization story. It is a **reproducibility** story.

| Factor | OpenRouter | Self-Hosted |
|--------|-----------|-------------|
| Model version control | No -- OpenRouter may update model weights silently | **Yes** -- pin exact HuggingFace revision |
| Quantization control | Unknown -- provider may quantize without disclosure | **Yes** -- choose FP16/INT8/INT4 explicitly |
| Inference parameters | Limited -- temperature, top_p via API; no control over batch scheduling | **Yes** -- full vLLM config (max_model_len, gpu_memory_utilization, etc.) |
| Rate limits | Yes -- OpenRouter applies per-minute/per-hour limits | **No** -- single-tenant, no throttling |
| Latency consistency | Variable -- shared infrastructure, geographic routing | **Consistent** -- single GPU, predictable latency |
| Reproducibility | Approximate -- same model name, unknown backend details | **Exact** -- same weights, same inference engine, same seed behavior |
| Cost predictability | Variable ($0.50-$5.00/experiment depending on token count) | **Fixed** -- pay for wall-clock time regardless of token count |
| Paper comparability | High (same model name, likely similar weights) | **Maximum** (provably identical model and weights) |

**The paper used Qwen3-8B.** OpenRouter serves Qwen3-8B. Self-hosting serves Qwen3-8B. The results should be comparable in all three cases. But "should be" is not "are." Self-hosting gives you **certainty** that the model is identical. OpenRouter gives you **probability** that it is.

For a paper reproduction study, certainty matters.

---

## Detailed Cost Scenarios

### Scenario A: Stick with OpenRouter (Status Quo)

| Item | Monthly Cost |
|------|-------------|
| OpenRouter API, 4 experiments | $2.00-$20.00 |
| GCE e2-medium spot (orchestration) | ~$0.19-$0.38/experiment = $0.76-$1.52 |
| GCS storage | <$1.00 |
| **Total** | **$3.76-$22.52** |

### Scenario B: Self-Host on GCP g2-standard-4 (L4) Spot

| Item | Monthly Cost |
|------|-------------|
| g2-standard-4 spot, 4x 24hr experiments | $27.08 |
| GCS storage (model cache PD) | ~$5.00 |
| GCS storage (results) | <$1.00 |
| **Total** | **~$33.08** |

### Scenario C: Self-Host on GCP n1-standard-2 + T4 Spot

| Item | Monthly Cost |
|------|-------------|
| n1-standard-2 + T4 spot, 4x 24hr experiments | $16.48 |
| GCS storage (model cache PD) | ~$5.00 |
| GCS storage (results) | <$1.00 |
| **Total** | **~$22.48** |

### Scenario D: Self-Host on AWS g6.xlarge (L4) Spot

| Item | Monthly Cost |
|------|-------------|
| g6.xlarge spot, 4x 24hr experiments | ~$16.32 |
| S3 storage | <$1.00 |
| **Total** | **~$17.32** |

### Comparison

| Scenario | Monthly Cost | % of Budget | Quantization | Reproducibility |
|----------|-------------|-------------|--------------|-----------------|
| A: OpenRouter | $3.76-$22.52 | 4-23% | Unknown | High (not guaranteed) |
| B: GCP L4 | ~$33.08 | 33% | FP16 (none) | **Maximum** |
| C: GCP T4 | ~$22.48 | 22% | INT8 (required) | High (minor quant difference) |
| D: AWS L4 | ~$17.32 | 17% | FP16 (none) | **Maximum** |

---

## What Experiment Volume Makes Self-Hosting Break Even?

### Fixed costs of self-hosting (monthly)
- Persistent disk with cached model: ~$5/month
- Operational overhead (your time setting up, debugging): Priceless (but real)

### Variable cost comparison

**GCP g2-standard-4 (L4) spot at $0.28/hr:**

| # Experiments (24hr each) | Self-Host Cost | OpenRouter Cost (at $2.50/exp) | OpenRouter Cost (at $5.00/exp) |
|--------------------------|----------------|-------------------------------|-------------------------------|
| 2 | $18.54 | $5.00 | $10.00 |
| 4 | $32.08 | $10.00 | $20.00 |
| 8 | $59.16 | $20.00 | $40.00 |
| 12 | $86.24 | $30.00 | $60.00 |
| 16 | $113.32 | $40.00 | $80.00 |

**Self-hosting on GCP L4 never breaks even vs. OpenRouter on pure cost** at any reasonable experiment volume within the $100 budget.

**GCP n1-standard-2 + T4 spot at ~$0.17/hr:**

| # Experiments (12hr each) | Self-Host Cost | OpenRouter Cost (at $5.00/exp) |
|--------------------------|----------------|-------------------------------|
| 2 | $9.08 | $10.00 |
| 4 | $13.16 | $20.00 |
| 8 | $21.32 | $40.00 |

**The T4 option breaks even at ~2 experiments/month IF your OpenRouter costs are consistently $5.00+/experiment.** Below $2.50/experiment on OpenRouter, self-hosting never wins on cost.

---

## Serving Architecture (If You Proceed)

### Recommended: g2-standard-4 (L4) with vLLM

```bash
# 1. Launch g2-standard-4 spot instance
gcloud compute instances create "gepa-qwen3-${EXPERIMENT_ID}" \
  --zone=us-central1-a \
  --machine-type=g2-standard-4 \
  --provisioning-model=SPOT \
  --instance-termination-action=STOP \
  --no-restart-on-failure \
  --service-account=gepa-runner@${PROJECT_ID}.iam.gserviceaccount.com \
  --scopes=cloud-platform \
  --metadata=experiment-id=${EXPERIMENT_ID} \
  --image-family=common-gpu-debian-12 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=50GB \
  --boot-disk-type=pd-balanced \
  --labels=project=gepa-mutations,experiment=${EXPERIMENT_ID}

# 2. Download model (~16GB, takes 3-5 min on GCE)
huggingface-cli download Qwen/Qwen3-8B --local-dir /opt/models/qwen3-8b

# 3. Serve with vLLM (FP16, no quantization needed on L4 24GB)
python -m vllm.entrypoints.openai.api_server \
  --model /opt/models/qwen3-8b \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 32768 \
  --host 0.0.0.0 --port 8000

# 4. Point GEPA runner at local endpoint
# Set LLM base URL to http://localhost:8000/v1
# Set model name to Qwen/Qwen3-8B
```

### Budget Option: n1-standard-2 + T4 with vLLM (INT8)

```bash
# 1. Launch n1-standard-2 + T4 spot instance
gcloud compute instances create "gepa-qwen3-t4-${EXPERIMENT_ID}" \
  --zone=us-central1-b \
  --machine-type=n1-standard-2 \
  --provisioning-model=SPOT \
  --instance-termination-action=STOP \
  --no-restart-on-failure \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --service-account=gepa-runner@${PROJECT_ID}.iam.gserviceaccount.com \
  --scopes=cloud-platform \
  --metadata=experiment-id=${EXPERIMENT_ID} \
  --image-family=common-gpu-debian-12 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=50GB \
  --boot-disk-type=pd-balanced \
  --labels=project=gepa-mutations,experiment=${EXPERIMENT_ID}

# 2. Download model
huggingface-cli download Qwen/Qwen3-8B --local-dir /opt/models/qwen3-8b

# 3. Serve with vLLM (INT8 quantization to fit T4 16GB)
python -m vllm.entrypoints.openai.api_server \
  --model /opt/models/qwen3-8b \
  --quantization int8 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --host 0.0.0.0 --port 8000

# Note: max-model-len reduced to 8192 to fit in T4 16GB with INT8.
# This limits context window but is sufficient for GEPA prompt generation.
```

### Self-Deletion (Both Options)

```bash
# MANDATORY: Add to startup script
cleanup() {
  gsutil -m rsync -r ./results/ \
    "gs://gepa-mutations-${PROJECT_ID}/experiments/${EXPERIMENT_ID}/results/" || true
  gcloud compute instances delete "${INSTANCE_NAME}" \
    --zone="${ZONE}" --quiet
}
trap cleanup EXIT
```

---

## Spot Eviction Handling

Qwen3-8B has a fast cold start (~5 minutes with cached PD). Spot eviction is manageable:

1. **Checkpoint GEPA state every iteration** to GCS (already implemented in the experiment runner)
2. **On eviction (30-second window):** Upload latest checkpoint to GCS
3. **On restart:** Pull checkpoint from GCS, reload model, resume from last iteration
4. **Model reloading:** 2-3 minutes for Qwen3-8B (fast due to small size)
5. **Total recovery time:** ~5-8 minutes (vs. 20-30+ minutes for larger models)

T4 spot eviction rates in us-central1 are historically low (<5%) because T4 demand has dropped. L4 eviction rates are slightly higher due to demand but still manageable.

---

## Final Verdict

| Question | Answer |
|----------|--------|
| Is self-hosting Qwen3-8B on GCP viable under $100/mo? | **Yes.** $16-$33/month for 4 experiments, well within budget. |
| Is it cheaper than OpenRouter? | **Only if OpenRouter experiments cost $5+/each.** At $0.50-$2.50/experiment, OpenRouter is cheaper. |
| Which GCP instance? | **g2-standard-4 (L4)** for reproducibility (FP16, no quant). **n1-standard-2 + T4** for lowest cost (INT8 required). |
| GCP or AWS? | **AWS g6.xlarge (L4) is 39% cheaper** at ~$0.17/hr spot vs. GCP's $0.28/hr for the same GPU. Use AWS if already on it. |
| Should you switch from OpenRouter? | **Only if reproducibility is a hard requirement.** The cost savings are marginal or negative. The value is in model control and scientific certainty. |
| What about the T4 option? | It is the cheapest ($0.14-0.17/hr) but requires INT8 quantization which introduces minor output differences. If exact FP16 reproduction is the goal, use L4 or better. |

### Concrete Recommendation

**If your OpenRouter experiments typically cost $0.50-$2.50 each:** Stay on OpenRouter. Self-hosting is more expensive after accounting for instance costs. The reproducibility argument is valid but may not justify 2-10x higher per-experiment cost.

**If your OpenRouter experiments cost $5.00+ each, OR if exact model reproducibility is a hard scientific requirement:** Self-host on AWS g6.xlarge (L4) spot at ~$0.17/hr. Same GPU as GCP's g2-standard-4 but 39% cheaper. Run Qwen3-8B at FP16 with no quantization. Monthly cost: ~$17 for 4x 24hr experiments.

**If you must use GCP specifically:** Use g2-standard-4 (L4) spot at $0.28/hr. FP16 Qwen3-8B with full context window. Monthly cost: ~$33 for 4x 24hr experiments. 33% of budget with plenty of headroom.

**Do not use the A100.** It costs 6.4x more than the L4 for a model that is 1/3 of the L4's VRAM capacity. There is no scenario where this makes sense for Qwen3-8B.

---

## Sources

- [GCP GPU Pricing](https://cloud.google.com/compute/gpus-pricing)
- [GCP Spot VM Pricing](https://cloud.google.com/spot-vms/pricing)
- [GCP VM Instance Pricing](https://cloud.google.com/compute/all-pricing)
- [g2-standard-4 - gcloud-compute.com](https://gcloud-compute.com/g2-standard-4.html)
- [g2-standard-4 - Economize](https://www.economize.cloud/resources/gcp/pricing/compute-engine/g2-standard-4/)
- [g2-standard-4 - Holori](https://calculator.holori.com/gcp/vm/g2-standard-4)
- [g2-standard-4 - Vantage](https://instances.vantage.sh/gcp/g2-standard-4)
- [g2-standard-4 - CloudPrice](https://cloudprice.net/gcp/compute/instances/g2-standard-4)
- [n1-standard-2 - Economize](https://www.economize.cloud/resources/gcp/pricing/compute-engine/n1-standard-2/)
- [n1-standard-4 - Economize](https://www.economize.cloud/resources/gcp/pricing/compute-engine/n1-standard-4/)
- [GPU Price Comparison 2026 - getdeploying.com](https://getdeploying.com/gpus)
- [NVIDIA L4 vs T4 - getdeploying.com](https://getdeploying.com/gpus/nvidia-l4-vs-nvidia-t4)
- [T4 vs L4 for Small Models - Clarifai](https://www.clarifai.com/blog/t4-vs-l4)
- [NVIDIA T4 vs L4 Cost-Efficiency - Oreate AI](https://www.oreateai.com/blog/nvidia-t4-vs-l4-decoding-the-costefficiency-for-your-ai-models/ba480720fd21f38aee04446c653b7155)
- [Benchmarking Qwen Models Across NVIDIA GPUs (T4, L4, H100) - Medium](https://medium.com/@wltsankalpa/benchmarking-qwen-models-across-nvidia-gpus-t4-l4-h100-architectures-finding-your-sweet-spot-a59a0adf9043)
- [Qwen3-8B Specifications - ApXML](https://apxml.com/models/qwen3-8b)
- [Deploy Qwen3 on GPU Cloud - Spheron](https://www.spheron.network/blog/deploy-qwen3-gpu-cloud/)
- [Qwen3 Speed Benchmark - Official](https://qwen.readthedocs.io/en/latest/getting_started/speed_benchmark.html)
- [GPU Cloud Pricing Comparison 2026 - Spheron](https://www.spheron.network/blog/gpu-cloud-pricing-comparison-2026/)
- [NVIDIA L4 GPU Price Guide 2026 - Jarvislabs](https://docs.jarvislabs.ai/blog/l4-gpu-price)
- [NVIDIA T4 Pricing 2026 - Fluence](https://www.fluence.network/blog/nvidia-t4/)
- [7 Cheapest Cloud GPU Providers 2026 - Northflank](https://northflank.com/blog/cheapest-cloud-gpu-providers)
- [Llama 3.1 8B vLLM Inference Performance - Microsoft](https://techcommunity.microsoft.com/blog/azurehighperformancecomputingblog/inference-performance-of-llama-3-1-8b-using-vllm-across-various-gpus-and-cpus/4448420)
- [NVIDIA T4 GPU Cost Analysis - Modal](https://modal.com/blog/nvidia-t4-price-article)
- [GCP GPU Pricing Comparison - Economize](https://www.economize.cloud/blog/gcp-gpu-pricing-comparison/)
- [AWS g6.xlarge - Vantage](https://instances.vantage.sh/aws/ec2/g6.xlarge)
- [AWS g5.xlarge - CloudPrice](https://cloudprice.net/aws/ec2/instances/g5.xlarge)
- [AWS g6.xlarge - Holori](https://calculator.holori.com/aws/ec2/g6.xlarge)
- [GCP Self-Hosting Cost Analysis (companion doc)](./gcp-self-hosting-cost-analysis.md)
- [AWS Self-Hosting Cost Analysis (companion doc)](./self-hosting-cost-analysis.md)
