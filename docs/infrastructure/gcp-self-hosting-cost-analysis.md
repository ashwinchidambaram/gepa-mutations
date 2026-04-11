# Self-Hosting LLMs on GCP GPU Instances: Cost Analysis

**Date:** 2026-03-22
**Budget Ceiling:** $100/month hard limit
**Current approach:** OpenRouter API for Qwen3-8B at $0.05/$0.40 per M tokens (in/out)
**Current cost per experiment:** $0.50-$5.00
**Experiment runtime:** 12-24 hours each
**Experiment volume:** 2-4 experiments/month

---

## Executive Summary

**Recommendation: Do NOT self-host on GCP. It is even more expensive than AWS for this workload.**

GCP GPU instances are uniformly more expensive than their AWS equivalents. The cheapest viable option for gpt-oss-20b (g2-standard-4 spot with 1x L4, $0.28/hr) costs $3.38-$6.76 per experiment -- slightly below the AWS g5.2xlarge spot option ($5.59-$11.18) but with a weaker GPU (L4 24GB vs A10G 24GB) that produces lower throughput. Qwen2.5-VL-72B is flatly impossible: the cheapest 4-GPU A100 instance (a2-highgpu-4g) costs $86.49-$172.98 per experiment on spot, exceeding the entire monthly budget on a single run.

GCP GPU instances cost 1.5-3x more than AWS equivalents at the A100 tier. The L4-class g2 instances are the one competitive option, but they offer the same 24GB VRAM as AWS g5/g6 instances at similar or higher prices.

**Stick with OpenRouter.** Neither GCP nor AWS self-hosting makes sense at this scale and budget.

---

## GCP GPU Instance Pricing Table

All prices are for **us-central1 (Iowa)** region, verified 2026-03-22.

### Accelerator-Optimized Instances

| Instance | GPUs | GPU Type | VRAM | vCPUs | RAM | On-Demand ($/hr) | Spot ($/hr) | Spot Savings |
|----------|------|----------|------|-------|-----|-----------------|-------------|-------------|
| g2-standard-4 | 1x L4 | NVIDIA L4 | 24 GB | 4 | 16 GB | $0.71 | $0.28 | 60% |
| g2-standard-8 | 1x L4 | NVIDIA L4 | 24 GB | 8 | 32 GB | $0.85 | $0.34 | 60% |
| g2-standard-12 | 1x L4 | NVIDIA L4 | 24 GB | 12 | 48 GB | $1.00 | $0.40 | 60% |
| a2-highgpu-1g | 1x A100-40GB | NVIDIA A100 | 40 GB | 12 | 85 GB | $3.67 | $1.80 | 51% |
| a2-highgpu-4g | 4x A100-40GB | NVIDIA A100 | 160 GB | 48 | 340 GB | $14.69 | $7.22 | 51% |
| a2-ultragpu-1g | 1x A100-80GB | NVIDIA A100 | 80 GB | 12 | 170 GB | $5.07 | $2.53 | 50% |
| a3-highgpu-8g | 8x H100-80GB | NVIDIA H100 | 640 GB | 208 | 1.9 TB | $88.49 | $81.53 | 8% |

### Committed Use Discounts (CUD) -- Where Available

| Instance | 1-Year CUD ($/mo) | 3-Year CUD ($/mo) | On-Demand ($/mo) | 1Y Savings | 3Y Savings |
|----------|-------------------|-------------------|------------------|-----------|-----------|
| g2-standard-4 | $325.07 | $232.19 | $515.99 | 37% | 55% |
| g2-standard-8 | $392.58 | $280.42 | $623.15 | 37% | 55% |
| g2-standard-12 | $460.09 | $328.64 | $730.30 | 37% | 55% |
| a2-highgpu-1g | $1,689.37 | $938.57 | $2,681.57 | 37% | 65% |
| a2-highgpu-4g | $6,757.48 | $3,754.27 | $10,726.28 | 37% | 65% |
| a2-ultragpu-1g | N/A | N/A | $3,700.22 | -- | -- |
| a3-highgpu-8g | N/A | N/A | $64,598 | -- | -- |

**Note on CUDs:** Committed use discounts are irrelevant for this project. They require 1-3 year commitments and minimum monthly spends of $232-$3,754. The entire monthly budget is $100. CUDs are for production workloads with consistent utilization, not a research project running 2-4 experiments per month.

**Note on Sustained Use Discounts (SUDs):** GCE automatically applies SUDs when an instance runs >25% of a month (~180 hours). Since experiments run 12-24 hours each, SUDs will never apply. Spot VMs do not receive SUDs regardless.

Sources: [gcloud-compute.com](https://gcloud-compute.com/a2-highgpu-1g.html), [Vantage](https://instances.vantage.sh/gcp/a2-ultragpu-1g), [Economize](https://www.economize.cloud/resources/gcp/pricing/compute-engine/a2-ultragpu-1g/), [GCP GPU Pricing](https://cloud.google.com/compute/gpus-pricing), [GCP Spot VM Pricing](https://cloud.google.com/spot-vms/pricing).

---

## Model 1: gpt-oss-20b (21B parameters, 3.6B active -- MoE architecture)

### VRAM Requirements

| Precision | VRAM Required | Notes |
|-----------|--------------|-------|
| FP16/BF16 | ~41 GB | All 21B params must be resident even though only 3.6B active |
| INT8 | ~21 GB | Fits on L4 24GB with KV cache headroom |
| MXFP4 | ~12-16 GB | Comfortably fits on L4 24GB |

### Instance Recommendations for gpt-oss-20b

| Instance | Total VRAM | Can Run gpt-oss-20b? | Quantization Required | Practical? |
|----------|------------|----------------------|----------------------|-----------|
| g2-standard-4 (1x L4) | 24 GB | Yes | INT8 or MXFP4 | **Best option** -- cheapest GPU |
| g2-standard-8 (1x L4) | 24 GB | Yes | INT8 or MXFP4 | Same GPU, more CPU/RAM |
| g2-standard-12 (1x L4) | 24 GB | Yes | INT8 or MXFP4 | Overkill on CPU side |
| a2-highgpu-1g (1x A100-40GB) | 40 GB | Yes | FP16 fits (tight), INT8 comfortable | **3.5x more expensive** for 1.7x more VRAM |
| a2-ultragpu-1g (1x A100-80GB) | 80 GB | Yes | No quant needed | Absurd overkill at $5.07/hr |

**Best GCP instance for gpt-oss-20b:** `g2-standard-4` (1x L4, 24GB, $0.28/hr spot). The model fits at INT8 (~21GB) with 3GB headroom for KV cache, or comfortably at MXFP4 (~16GB) with 8GB headroom. The extra CPU/RAM on g2-standard-8 and g2-standard-12 is unnecessary -- the workload is GPU-bound, not CPU-bound.

The a2-highgpu-1g is NOT justified. Paying $1.80/hr instead of $0.28/hr (6.4x more) for the A100 gives you higher throughput per token, but the experiment is already running for 12-24 hours. Higher throughput does not reduce experiment duration proportionally because the bottleneck shifts to scoring and other pipeline stages.

### Cost Per Experiment (gpt-oss-20b on GCP)

Using g2-standard-4 spot (1x L4, $0.28/hr):

| Scenario | Hours | On-Demand Cost | Spot Cost |
|----------|-------|---------------|-----------|
| 12-hour experiment | 12 | $8.48 | $3.38 |
| 24-hour experiment | 24 | $16.96 | $6.76 |
| + Startup overhead (~30 min) | +0.5 | +$0.35 | +$0.14 |

Using a2-highgpu-1g spot (1x A100-40GB, $1.80/hr) -- for comparison:

| Scenario | Hours | On-Demand Cost | Spot Cost |
|----------|-------|---------------|-----------|
| 12-hour experiment | 12 | $44.04 | $21.65 |
| 24-hour experiment | 24 | $88.08 | $43.30 |

**Startup overhead includes:** instance boot (~2-3 min), model download from HuggingFace (~10-15 min for 41GB on GCE, slightly slower than AWS due to less consistent network to HF CDN), vLLM server startup + model loading (~5-10 min). Total: ~20-30 minutes before first token served.

### Monthly Cost (gpt-oss-20b on GCP)

**g2-standard-4 spot:**

| Experiments/Month | Spot Cost (12hr each) | Spot Cost (24hr each) |
|------------------|----------------------|----------------------|
| 2 | $7.04 | $13.80 |
| 3 | $10.56 | $20.70 |
| 4 | $14.08 | $27.60 |

**a2-highgpu-1g spot:**

| Experiments/Month | Spot Cost (12hr each) | Spot Cost (24hr each) |
|------------------|----------------------|----------------------|
| 2 | $43.58 | $86.88 |
| 3 | $65.37 | $130.32 |
| 4 | **$87.16** | **$173.76** |

**COST WARNING:** The g2-standard-4 spot stays within budget at all scenarios ($7-$28/month). But this is misleading -- you are paying $3.38-$6.76 per experiment for self-hosted gpt-oss-20b inference vs. $0.50-$5.00 per experiment on OpenRouter for Qwen3-8B. They are different models. The question is whether gpt-oss-20b on L4 provides enough value over Qwen3-8B on OpenRouter to justify the infrastructure overhead.

The a2-highgpu-1g exceeds budget at 4 experiments/month of 24 hours each ($174). Even at 12 hours, 4 experiments is $87 -- tight against the $100 ceiling with zero margin for other costs.

### Estimated Throughput (gpt-oss-20b)

| Instance | GPU | Quantization | Est. Tokens/sec | Source |
|----------|-----|-------------|-----------------|--------|
| g2-standard-4 (1x L4) | L4 24GB | MXFP4/INT8 | ~80-200 | DevForth benchmarks (L4 for gpt-oss-20b) |
| a2-highgpu-1g (1x A100-40GB) | A100 40GB | FP16 | ~9,743 | GPUStack benchmark |
| Reference: A100 80GB (vLLM) | A100 80GB | FP16 | ~10,920 | GPUStack (async scheduling) |

The L4 throughput for gpt-oss-20b is dramatically lower than A100. The L4 has 30.3 TFLOPS FP16 and 395 GB/s memory bandwidth vs the A100's 312 TFLOPS and 2,039 GB/s. That is roughly 10x less compute and 5x less bandwidth. The ~80-200 tok/s estimate on L4 is based on the DevForth benchmark data showing L4 at "approximately 80 tokens/s for short sequences" with gpt-oss-20b.

At 80-200 tok/s, a workload that processes millions of tokens will take significantly longer than on OpenRouter (which scales horizontally). An experiment that takes 12 hours on OpenRouter could take 24+ hours on a single L4.

---

## Model 2: Qwen2.5-VL-72B-Instruct (72B parameters)

### VRAM Requirements

| Precision | VRAM Required | Min GPUs (A100-40GB) | Min GPUs (A100-80GB) |
|-----------|--------------|---------------------|---------------------|
| FP16/BF16 | ~144 GB | 4x (160GB total) | 2x (160GB total) |
| FP8 | ~72 GB | 2x (80GB total) | 1x (80GB) -- very tight |
| INT4 (AWQ/GPTQ) | ~37 GB (text-only) | 1x (40GB) -- text only | 1x (80GB) |
| VL variant overhead | up to 384 GB | N/A -- see below | N/A -- see below |

**Critical: This is a vision-language model.** The Qwen2.5-**VL**-72B has a vision encoder on top of the 72B text model. The Qwen documentation states deployment can require up to 384 GB for full precision. Even at INT4, the VL variant is significantly more memory-hungry than the text-only Qwen2.5-72B due to the image processing pipeline.

### Instance Recommendations for Qwen2.5-VL-72B

| Instance | Total VRAM | Can Run Qwen2.5-VL-72B? | Notes |
|----------|------------|--------------------------|-------|
| g2-standard-4 (1x L4) | 24 GB | **No.** | Not even INT4 text-only fits with overhead |
| a2-highgpu-1g (1x A100-40GB) | 40 GB | **No.** | INT4 text-only might fit; VL will not |
| a2-ultragpu-1g (1x A100-80GB) | 80 GB | **Barely** at INT4 | INT4 text-only; VL variant needs more |
| a2-highgpu-4g (4x A100-40GB) | 160 GB | **Yes** at FP16 | But only 160GB, tight with KV cache; TP=4 across A100s via NVLink |
| a3-highgpu-8g (8x H100-80GB) | 640 GB | **Yes, comfortably** | But at $81-88/hr, this is a joke at our budget |

**Realistic option:** a2-highgpu-4g (4x A100-40GB, 160GB total). This can run Qwen2.5-VL-72B at FP16 with tensor parallelism across 4 GPUs. The 160GB ceiling is tight for the VL variant -- `max_model_len` must be reduced from the default 32768 to avoid OOM with KV cache. Benchmarks show 154.56 tok/s total throughput on 4x A100-40GB for this model with 50 concurrent requests.

GCP does not offer an equivalent to AWS's g5.48xlarge (8x A10G, 192GB, no NVLink) at a reasonable price point. The a2-highgpu-4g at $7.22/hr spot is the cheapest viable option. There is no budget-friendly path.

### Pricing Table for Qwen2.5-VL-72B Viable Instances

| Instance | GPUs | On-Demand ($/hr) | Spot ($/hr) | Spot Savings |
|----------|------|-----------------|-------------|-------------|
| a2-highgpu-4g | 4x A100-40GB | $14.69 | $7.22 | 51% |
| a2-ultragpu-1g | 1x A100-80GB | $5.07 | $2.53 | 50% |
| a3-highgpu-8g | 8x H100-80GB | $88.49 | $81.53 | 8% |

**Note:** The a2-ultragpu-1g (1x A100-80GB) can only run the INT4-quantized text-only version, and even then performance will be poor with no tensor parallelism for the VL variant. The a3-highgpu-8g's spot discount is a pathetic 8% -- H100 spot pricing on GCP barely discounts at all due to high demand.

### Cost Per Experiment (Qwen2.5-VL-72B)

Using a2-highgpu-4g spot (4x A100-40GB, $7.22/hr):

| Scenario | Hours | On-Demand Cost | Spot Cost |
|----------|-------|---------------|-----------|
| 12-hour experiment | 12 | $176.28 | $86.64 |
| 24-hour experiment | 24 | $352.56 | $173.28 |
| + Startup overhead (~45 min) | +0.75 | +$11.02 | +$5.42 |

Using a3-highgpu-8g spot (8x H100-80GB, $81.53/hr):

| Scenario | Hours | On-Demand Cost | Spot Cost |
|----------|-------|---------------|-----------|
| 12-hour experiment | 12 | $1,061.88 | $978.36 |
| 24-hour experiment | 24 | $2,123.76 | $1,956.72 |

### Monthly Cost (Qwen2.5-VL-72B)

**a2-highgpu-4g spot:**

| Experiments/Month | Spot Cost (12hr each) | Spot Cost (24hr each) |
|------------------|----------------------|----------------------|
| 1 | $92.06 | $178.70 |
| 2 | $184.12 | $357.40 |

**COST CRITICAL:** A single 12-hour experiment on the cheapest viable instance costs $87-$92 on spot -- nearly the entire $100/month budget. A 24-hour experiment costs $173-$179 -- 1.7x the budget. Two experiments per month is $357 -- 3.6x the budget. The H100 instances are not even worth discussing at $978+ per experiment.

### Estimated Throughput (Qwen2.5-VL-72B)

| Instance | Quantization | Throughput (tok/s) | Source |
|----------|-------------|-------------------|--------|
| a2-highgpu-4g (4x A100-40GB) | FP16, TP=4 | ~155 | DatabaseMart benchmark |
| a2-highgpu-4g (4x A100-40GB) | INT4, TP=4 | Tensor dimension alignment issues | vLLM GitHub #12988 |
| a3-highgpu-8g (8x H100-80GB) | FP16, TP=8 | ~400-800 (est.) | Extrapolated |
| Reference: 4x A6000 (192GB) | FP16, TP=4 | ~450 | DatabaseMart benchmark |

**Important:** The 4x A100-40GB configuration shows only 155 tok/s for Qwen2.5-VL-72B because the 160GB total VRAM is tight. The benchmark data shows 4x A6000 (192GB total) achieves nearly 3x higher throughput (450 tok/s) for the same model, demonstrating that memory bottleneck cripples performance on the A100-40GB config. This is 160GB working for you, not 320GB like the p4d.24xlarge (8x A100-40GB) on AWS.

**INT4 quantization is problematic.** The vLLM GitHub issues document dimension alignment problems with INT4 GPTQ-quantized Qwen2.5-VL-72B that restrict tensor parallelism to TP=1 (single GPU). This makes INT4 quantization impractical on multi-GPU setups for this specific model.

---

## Break-Even Analysis

### gpt-oss-20b: GCP Self-Hosting vs. OpenRouter

| OpenRouter Cost/Experiment | Self-Host Spot Cost (g2-standard-4, 24hr) | Ratio | Break-Even? |
|---------------------------|------------------------------------------|-------|-------------|
| $0.50 | $6.90 | 13.8x | **Never** |
| $2.50 | $6.90 | 2.8x | **Never** |
| $5.00 | $6.90 | 1.4x | **Never** |

At 12-hour experiments ($3.52 spot):

| OpenRouter Cost/Experiment | Self-Host Spot Cost (g2-standard-4, 12hr) | Ratio | Break-Even? |
|---------------------------|------------------------------------------|-------|-------------|
| $0.50 | $3.52 | 7.0x | **Never** |
| $2.50 | $3.52 | 1.4x | **Never** |
| $5.00 | $3.52 | 0.7x | **Self-host wins IF you need gpt-oss-20b** |

**Key insight:** If your OpenRouter experiments consistently cost $5.00 each, AND you specifically need gpt-oss-20b (not available on OpenRouter), then 12-hour experiments on a g2-standard-4 spot become cheaper per-experiment at $3.52. But you are comparing apples (Qwen3-8B on OpenRouter) to oranges (gpt-oss-20b on L4). The break-even only works if gpt-oss-20b is a hard requirement.

### Qwen2.5-VL-72B: Break-Even Is Impossible

| OpenRouter Cost/Experiment | Self-Host Spot Cost (a2-highgpu-4g, 24hr) | Ratio | Break-Even? |
|---------------------------|-------------------------------------------|-------|-------------|
| $0.50 | $178.70 | 357x | **Never** |
| $5.00 | $178.70 | 35.7x | **Never** |
| $50.00 (hypothetical) | $178.70 | 3.6x | **Never** at this budget |

### What Experiment Volume Would Make Self-Hosting Break Even?

For gpt-oss-20b on g2-standard-4, running the instance **continuously** (730 hours/month):

- g2-standard-4 spot, 730 hrs: $204.40/month
- At 30 experiments/month (24hr each): $6.81/experiment
- At 60 experiments/month (12hr each): $3.41/experiment

This is 2x the budget. Even continuous operation does not justify self-hosting within $100/month.

---

## Budget Reality Check

| Scenario | Cost | % of $100 Budget | Verdict |
|----------|------|-----------------|---------|
| OpenRouter, 4 experiments | $2-$20 | 2-20% | Well within budget |
| gpt-oss-20b, g2-standard-4 spot, 1x 12hr | $3.52 | 3.5% | Fits, comparable to OpenRouter |
| gpt-oss-20b, g2-standard-4 spot, 4x 24hr | $27.60 | 27.6% | Fits, but with operational overhead |
| gpt-oss-20b, a2-highgpu-1g spot, 4x 24hr | $173.76 | **173.8%** | Over budget |
| Qwen2.5-VL-72B, a2-highgpu-4g spot, 1x 12hr | $92.06 | **92.1%** | Nearly the entire budget for ONE run |
| Qwen2.5-VL-72B, a2-highgpu-4g spot, 1x 24hr | $178.70 | **178.7%** | 1.8x budget for ONE run |
| Qwen2.5-VL-72B, a3-highgpu-8g spot, 1x 12hr | $978.36 | **978.4%** | Nearly 10x the entire budget |

---

## Startup Time Estimates

| Model | Instance | Steps | Est. Total Time |
|-------|----------|-------|----------------|
| gpt-oss-20b | g2-standard-4 (L4) | Boot (2min) + download 41GB (12-18min) + vLLM load (5-10min) | **19-30 min** |
| gpt-oss-20b | g2-standard-4 (cached PD) | Boot (2min) + vLLM load (5-10min) | **7-12 min** |
| Qwen2.5-VL-72B | a2-highgpu-4g | Boot (3min) + download 144GB (25-45min) + vLLM TP init (10-15min) | **38-63 min** |
| Qwen2.5-VL-72B | a2-highgpu-4g (cached PD) | Boot (3min) + vLLM TP load (10-15min) | **13-18 min** |

**Note on GCP model caching:** Use a persistent disk snapshot with the model pre-downloaded to avoid the download step. A 100GB Balanced PD costs ~$10/month. A 200GB PD for the 72B model costs ~$20/month. This saves 12-45 minutes per experiment start but adds to monthly cost.

---

## Serving Architecture (If You Proceeded Anyway)

### gpt-oss-20b on g2-standard-4

```bash
# 1. Launch g2-standard-4 spot instance
gcloud compute instances create "gepa-gpu-${EXPERIMENT_ID}" \
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
  --boot-disk-size=100GB \
  --boot-disk-type=pd-balanced \
  --labels=project=gepa-mutations,experiment=${EXPERIMENT_ID}

# 2. SSH in and download model
huggingface-cli download openai/gpt-oss-20b --local-dir /opt/models/gpt-oss-20b

# 3. Serve with vLLM (INT8 quantization to fit L4 24GB)
python -m vllm.entrypoints.openai.api_server \
  --model /opt/models/gpt-oss-20b \
  --quantization int8 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 16384 \
  --host 0.0.0.0 --port 8000

# 4. Point GEPA runner at local endpoint
# Set model endpoint to http://localhost:8000/v1
```

### Qwen2.5-VL-72B on a2-highgpu-4g

```bash
# 1. Launch a2-highgpu-4g spot instance
gcloud compute instances create "gepa-gpu-72b-${EXPERIMENT_ID}" \
  --zone=us-central1-a \
  --machine-type=a2-highgpu-4g \
  --provisioning-model=SPOT \
  --instance-termination-action=STOP \
  --no-restart-on-failure \
  --service-account=gepa-runner@${PROJECT_ID}.iam.gserviceaccount.com \
  --scopes=cloud-platform \
  --image-family=common-gpu-debian-12 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=300GB \
  --boot-disk-type=pd-balanced \
  --labels=project=gepa-mutations,experiment=${EXPERIMENT_ID}

# 2. Download model
huggingface-cli download Qwen/Qwen2.5-VL-72B-Instruct \
  --local-dir /opt/models/qwen2.5-vl-72b

# 3. Serve with vLLM (tensor parallel across 4 A100s)
python -m vllm.entrypoints.openai.api_server \
  --model /opt/models/qwen2.5-vl-72b \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --host 0.0.0.0 --port 8000

# Note: max-model-len reduced to 8192 from default 32768
# to fit in 160GB total VRAM with KV cache overhead.
# This limits context window significantly.
```

---

## GCP vs AWS Comparison

### gpt-oss-20b (Best Instance Each Cloud)

| Metric | AWS g5.2xlarge (1x A10G) | GCP g2-standard-4 (1x L4) | Winner |
|--------|------------------------|---------------------------|--------|
| GPU | A10G 24GB | L4 24GB | Tie (same VRAM) |
| GPU Compute (FP16) | 31.2 TFLOPS | 30.3 TFLOPS | Tie |
| Memory Bandwidth | 600 GB/s | 395 GB/s | **AWS** (1.5x) |
| On-Demand $/hr | $1.212 | $0.71 | **GCP** (41% cheaper) |
| Spot $/hr | $0.466 | $0.28 | **GCP** (40% cheaper) |
| Spot 24hr cost | $11.18 | $6.76 | **GCP** |
| Spot 12hr cost | $5.59 | $3.38 | **GCP** |
| vCPUs / RAM | 8 / 32 GB | 4 / 16 GB | **AWS** (more headroom) |
| Est. Throughput | ~500-1,500 tok/s | ~80-200 tok/s | **AWS** (L4 bandwidth bottleneck) |
| Spot Interruption | Variable | Variable | Tie |

**Verdict for gpt-oss-20b:** GCP is cheaper per hour, but the L4's lower memory bandwidth (395 vs 600 GB/s) means significantly lower throughput. The A10G on AWS will complete the inference workload faster, potentially making the total cost similar. If experiments are time-boxed (e.g., always 24 hours), GCP's g2-standard-4 saves ~$4.42/experiment on spot. If the experiment runs until completion (variable time), the A10G's higher throughput may mean fewer hours needed.

However, GCP's g2-standard-4 has only 4 vCPUs and 16GB RAM -- half of the g5.2xlarge. If the GEPA runner itself needs more than 16GB RAM (unlikely for API orchestration, but possible with large datasets in memory), you need the g2-standard-8 at $0.34/hr spot, still cheaper than AWS.

### Qwen2.5-VL-72B (Best Instance Each Cloud)

| Metric | AWS p4d.24xlarge (8x A100-40GB) | AWS g5.48xlarge (8x A10G) | GCP a2-highgpu-4g (4x A100-40GB) |
|--------|-------------------------------|--------------------------|----------------------------------|
| GPUs | 8x A100-40GB | 8x A10G | 4x A100-40GB |
| Total VRAM | 320 GB | 192 GB | 160 GB |
| NVLink | Yes | No | Yes |
| Spot $/hr | $10.70 | $6.88 | $7.22 |
| Spot 24hr cost | $256.80 | $165.14 | $173.28 |
| Throughput (est.) | 800-1,500 tok/s | 200-500 tok/s | ~155 tok/s |

**Verdict for Qwen2.5-VL-72B:** All options are absurdly over budget. The GCP a2-highgpu-4g is the worst choice despite being mid-range in price, because 160GB VRAM is barely enough and throughput is crippled (155 tok/s). AWS g5.48xlarge is cheapest but has no NVLink. AWS p4d.24xlarge has the best VRAM headroom (320GB) and NVLink but costs the most. None of them matter -- they all exceed the $100 budget on a single experiment.

### Head-to-Head Summary

| Factor | GCP | AWS | Winner |
|--------|-----|-----|--------|
| Cheapest gpt-oss-20b spot | $0.28/hr (g2-standard-4) | $0.39/hr (g5.xlarge) | **GCP** (28% cheaper) |
| Best gpt-oss-20b throughput | ~80-200 tok/s (L4) | ~500-1,500 tok/s (A10G) | **AWS** (5-7x faster) |
| Cheapest 72B viable spot | $7.22/hr (a2-highgpu-4g) | $6.88/hr (g5.48xlarge) | **AWS** (5% cheaper) |
| Best 72B throughput | ~155 tok/s (4x A100-40GB) | ~800-1,500 tok/s (8x A100-40GB) | **AWS** (5-10x faster) |
| Spot discount depth | 50-60% | 51-87% | **AWS** (deeper on some instances) |
| H100 spot discount | 8% (a3-highgpu-8g) | N/A (p5 availability) | Neither (both expensive) |
| CUD availability | g2 and a2-highgpu only | All instance types | **AWS** (broader RI coverage) |
| Deep Learning AMI | Yes (deeplearning-platform-release) | Yes (Deep Learning AMI) | Tie |

---

## What Would Need to Change for GCP Self-Hosting to Make Sense

1. **Budget increase to $200+/month** -- This opens up gpt-oss-20b on g2-standard-4 spot at 4-8 experiments/month with margin. The 72B model remains out of reach.

2. **Budget increase to $500+/month** -- This enables 2-3 experiments/month with Qwen2.5-VL-72B on a2-highgpu-4g spot.

3. **Experiment volume increase to 20+/month** -- At high volume with continuous instance usage, sustained use discounts kick in (~30% off on-demand after 25% of month). But spot VMs do not get SUDs.

4. **gpt-oss-20b is a hard requirement** -- If the research demands this specific model and OpenRouter does not offer it, self-hosting on GCP g2-standard-4 spot at $3.38-$6.76/experiment is defensible. It fits within budget and is 28% cheaper than AWS g5.xlarge spot.

5. **GCP is already your primary cloud** -- If you already have GCP projects, IAM, networking set up, the marginal operational cost of adding GPU instances is lower than setting up AWS from scratch (or vice versa). This project currently uses AWS.

---

## Final Verdict

| Question | Answer |
|----------|--------|
| Is GCP self-hosting gpt-oss-20b viable under $100/mo? | Technically yes at $14-$28/month, but lower throughput than AWS |
| Is GCP self-hosting Qwen2.5-VL-72B viable under $100/mo? | **No.** A single 12hr experiment costs $87-$92 -- the entire budget |
| Is GCP cheaper than AWS for gpt-oss-20b? | **Yes per-hour** (28% cheaper spot). **No per-token** (5-7x slower throughput) |
| Is GCP cheaper than AWS for Qwen2.5-VL-72B? | **No.** GCP a2-highgpu-4g ($7.22/hr, 160GB) vs AWS g5.48xlarge ($6.88/hr, 192GB) |
| Should we switch from OpenRouter to GCP self-hosting? | **No.** |
| Which cloud is better for self-hosting IF budget increased? | **AWS** for both models (better throughput, more instance variety, deeper spot discounts) |

**Stick with OpenRouter.** The conclusion is the same as the AWS analysis, but stronger: GCP GPU instances are generally more expensive than AWS for A100-class hardware, and the L4 instances are cheaper per hour but deliver dramatically less throughput. The $0.50-$5.00 per experiment cost on OpenRouter with zero infrastructure overhead remains the optimal choice at this scale and budget.

**If you must self-host gpt-oss-20b and are choosing between clouds:** AWS g5.2xlarge spot ($0.47/hr, A10G, better throughput) is the better pick despite higher hourly cost, because the A10G's 1.5x memory bandwidth advantage means experiments complete faster. GCP g2-standard-4 spot ($0.28/hr, L4) is cheaper per hour but slower, making total cost roughly equivalent once you account for longer experiment duration.

---

## Sources

- [GCP GPU Pricing](https://cloud.google.com/compute/gpus-pricing)
- [GCP Spot VM Pricing](https://cloud.google.com/spot-vms/pricing)
- [GCP Accelerator-Optimized VM Pricing](https://cloud.google.com/products/compute/pricing/accelerator-optimized)
- [g2-standard-4 - gcloud-compute.com](https://gcloud-compute.com/g2-standard-4.html)
- [g2-standard-8 - gcloud-compute.com](https://gcloud-compute.com/g2-standard-8.html)
- [g2-standard-12 - gcloud-compute.com](https://gcloud-compute.com/g2-standard-12.html)
- [a2-highgpu-1g - gcloud-compute.com](https://gcloud-compute.com/a2-highgpu-1g.html)
- [a2-highgpu-4g - gcloud-compute.com](https://gcloud-compute.com/a2-highgpu-4g.html)
- [a2-ultragpu-1g - gcloud-compute.com](https://gcloud-compute.com/a2-ultragpu-1g.html)
- [a2-ultragpu-1g - Vantage](https://instances.vantage.sh/gcp/a2-ultragpu-1g)
- [a2-ultragpu-1g - Economize](https://www.economize.cloud/resources/gcp/pricing/compute-engine/a2-ultragpu-1g/)
- [GCP Committed Use Discounts](https://docs.cloud.google.com/compute/docs/instances/signing-up-committed-use-discounts)
- [g5.xlarge AWS - Vantage](https://instances.vantage.sh/aws/ec2/g5.xlarge)
- [g6.xlarge AWS - Vantage](https://instances.vantage.sh/aws/ec2/g6.xlarge)
- [GPT-OSS-20B Hardware Requirements - IntuitionLabs](https://intuitionlabs.ai/articles/hardware-requirements-gpt-oss-20b)
- [GPT-OSS-20B Specs - ApXML](https://apxml.com/models/gpt-oss-20b)
- [GPT-OSS-20B A100 Throughput - GPUStack](https://docs.gpustack.ai/2.0/performance-lab/gpt-oss-20b/a100/)
- [GPT-OSS-20B L4/L40S/H100 Benchmark - DevForth](https://devforth.io/insights/self-hosted-gpt-real-response-time-token-throughput-and-cost-on-l4-l40s-and-h100-for-gpt-oss-20b/)
- [Qwen2.5-VL-72B VRAM Needs - Novita AI](https://blogs.novita.ai/qwen2-5-vl-72b-vram/)
- [Qwen2.5 Speed Benchmark - Official](https://qwen.readthedocs.io/en/v2.5/benchmark/speed_benchmark.html)
- [4x A100 vs 4x A6000 vLLM Benchmark - DatabaseMart](https://www.databasemart.com/blog/vllm-gpu-benchmark-a100-40gb-4)
- [Dual A100 vLLM Benchmark - DatabaseMart](https://www.databasemart.com/blog/vllm-gpu-benchmark-dual-a100-40gb)
- [vLLM Qwen2.5-VL Recipe](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen2.5-VL.html)
- [Qwen2.5-VL-72B TP Issues - vLLM GitHub #12988](https://github.com/vllm-project/vllm/issues/12988)
- [H100 Pricing Comparison 2026 - IntuitionLabs](https://intuitionlabs.ai/articles/h100-rental-prices-cloud-comparison)
- [AWS Self-Hosting Cost Analysis (companion doc)](../docs/self-hosting-cost-analysis.md)
