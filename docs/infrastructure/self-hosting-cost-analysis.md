# Self-Hosting LLMs on AWS GPU Instances: Cost Analysis

**Date:** 2026-03-22
**Budget Ceiling:** $100/month hard limit
**Current approach:** OpenRouter API for Qwen3-8B at $0.05/$0.40 per M tokens (in/out)
**Current cost per experiment:** $0.50-$5.00
**Experiment runtime:** 12-24 hours each
**Experiment volume:** 2-4 experiments/month

---

## Executive Summary

**Recommendation: Do NOT self-host. It is not close to viable at this budget and scale.**

Self-hosting either model blows through the $100/month budget on a single experiment. The cheapest viable option (gpt-oss-20b on a g5.2xlarge spot instance) costs $5.60-$11.18 per experiment -- comparable to the current OpenRouter cost ceiling, but with massive operational overhead, startup latency, spot interruption risk, and zero cost savings. The 72B model is flatly impossible under a $100/month budget. Stick with OpenRouter.

---

## Model 1: gpt-oss-20b (21B parameters, 3.6B active -- MoE architecture)

### Key Insight: This Is a Sparse MoE Model

gpt-oss-20b has 21B total parameters but only 3.6B active parameters (Mixture of Experts). This dramatically reduces VRAM requirements:
- **FP16:** ~41GB (full weights must be resident, even if only 3.6B active)
- **INT8:** ~21GB
- **MXFP4 (official quantization):** ~16GB -- fits on a single A10G 24GB GPU

### Instance Recommendations

| Instance | GPUs | GPU Type | Total VRAM | Can Run gpt-oss-20b? | Quantization Required |
|----------|------|----------|------------|----------------------|----------------------|
| g5.xlarge | 1x A10G | A10G | 24 GB | Yes | MXFP4 or INT8 |
| g5.2xlarge | 1x A10G | A10G | 24 GB | Yes | MXFP4 or INT8 |
| g5.4xlarge | 1x A10G | A10G | 24 GB | Yes | MXFP4 or INT8 |
| g5.12xlarge | 4x A10G | A10G | 96 GB | Yes (overkill) | FP16 fits, no quant needed |
| p3.2xlarge | 1x V100 | V100 | 16 GB | Barely | MXFP4 only, very tight |

**Best instance for gpt-oss-20b:** g5.2xlarge (1x A10G, 24GB). The model fits at INT8 (~21GB) or comfortably at MXFP4 (~16GB). The g5.xlarge has the same GPU but less CPU/RAM; g5.2xlarge gives headroom for vLLM's KV cache. The p3.2xlarge's 16GB V100 is too tight. The g5.12xlarge is 4.7x the cost for no GPU benefit on this model.

### Pricing Table

| Instance | On-Demand ($/hr) | Spot ($/hr) | Spot Savings |
|----------|-----------------|-------------|-------------|
| g5.xlarge | $1.006 | ~$0.39 | ~61% |
| g5.2xlarge | $1.212 | $0.466 | 62% |
| g5.4xlarge | $1.624 | $0.677 | 58% |
| g5.12xlarge | $5.672 | $2.588 | 54% |
| p3.2xlarge | $3.060 | $0.398 | 87% |

Source: instances.vantage.sh, verified 2026-03-22, us-east-1 region.

### Cost Per Experiment (gpt-oss-20b)

Using g5.2xlarge (the practical choice):

| Scenario | Hours | On-Demand Cost | Spot Cost |
|----------|-------|---------------|-----------|
| 12-hour experiment | 12 | $14.54 | $5.59 |
| 24-hour experiment | 24 | $29.09 | $11.18 |
| + Startup overhead (~30 min) | +0.5 | +$0.61 | +$0.23 |

**Startup overhead includes:** instance boot (~2 min), model download from HuggingFace (~10-15 min for 41GB), vLLM server startup + model loading (~5-10 min). Total: ~20-30 minutes before first token served.

### Monthly Cost (gpt-oss-20b)

| Experiments/Month | On-Demand (24hr each) | Spot (24hr each) |
|------------------|-----------------------|------------------|
| 2 | $59.38 | $22.82 |
| 3 | $89.07 | $34.23 |
| 4 | $118.76 | $45.64 |

**COST WARNING:** At 4 experiments/month on-demand, you exceed the $100 budget. Spot pricing keeps you under budget for gpt-oss-20b at 2-4 experiments/month, but this ignores a critical problem: **you are paying $5.59-$11.18 per experiment for self-hosted inference vs. $0.50-$5.00 per experiment on OpenRouter for Qwen3-8B.** Self-hosting gpt-oss-20b is a different model, not a cheaper way to run the same model.

### Estimated Throughput (gpt-oss-20b on A10G)

Based on benchmarks of similar MoE models on A10G hardware:
- **MXFP4 quantization, single A10G:** ~500-1,500 tokens/sec (estimated; A100 achieves ~9,700 tok/s, A10G is roughly 3-6x slower due to less memory bandwidth and compute)
- **For context:** OpenRouter's Qwen3-8B throughput is effectively unlimited (API scales horizontally)

The throughput difference matters: self-hosting a single GPU means serial inference. OpenRouter parallelizes across their fleet. Experiment wall-clock time could increase significantly.

---

## Model 2: Qwen2.5-VL-72B-Instruct (72B parameters)

### VRAM Requirements

| Precision | VRAM Required | Notes |
|-----------|--------------|-------|
| FP16/BF16 | ~144 GB | Minimum 4x A100-40GB or 2x A100-80GB |
| FP8 | ~72 GB | 2x A100-40GB or 1x A100-80GB |
| INT4 (AWQ/GPTQ) | ~37 GB | 2x A10G (48GB) with tensor parallelism |
| Vision-Language overhead | up to 384 GB | VL variant needs extra memory for image processing |

**Critical caveat:** The Qwen2.5-**VL**-72B variant (vision-language) demands significantly more memory than text-only Qwen2.5-72B due to the vision encoder. The 384GB figure for full deployment is documented by the Qwen team. Even at INT4, the VL model needs substantially more than the 37GB text-only figure.

### Instance Recommendations

| Instance | GPUs | GPU Type | Total VRAM | Can Run Qwen2.5-VL-72B? | Quantization |
|----------|------|----------|------------|--------------------------|-------------|
| g5.12xlarge | 4x A10G | A10G | 96 GB | Maybe at INT4, tight | INT4 required, no NVLink = slow TP |
| g5.48xlarge | 8x A10G | A10G | 192 GB | Yes at INT4/INT8 | INT4/INT8, but no NVLink = poor TP scaling |
| p3.8xlarge | 4x V100 | V100 | 64 GB | No | Even INT4 (~37GB text) won't fit VL variant |
| p4d.24xlarge | 8x A100 | A100-40GB | 320 GB | Yes, comfortably | FP16 or INT8, NVLink for fast TP |

**Reality:** The p4d.24xlarge is the only instance that runs this model well. The g5.48xlarge could work with INT4 quantization but A10G GPUs lack NVLink, making tensor parallelism across 8 GPUs painfully slow. The p3.8xlarge does not have enough VRAM.

### Pricing Table

| Instance | On-Demand ($/hr) | Spot ($/hr) | Spot Savings |
|----------|-----------------|-------------|-------------|
| g5.12xlarge | $5.672 | $2.588 | 54% |
| g5.48xlarge | $16.288 | $6.881 | 58% |
| p3.8xlarge | $12.240 | $1.716 | 86% |
| p4d.24xlarge | $21.960 | $10.700 | 51% |

Source: instances.vantage.sh, verified 2026-03-22, us-east-1 region.

### Cost Per Experiment (Qwen2.5-VL-72B)

Using p4d.24xlarge (the only realistic choice for quality inference):

| Scenario | Hours | On-Demand Cost | Spot Cost |
|----------|-------|---------------|-----------|
| 12-hour experiment | 12 | $263.52 | $128.40 |
| 24-hour experiment | 24 | $527.04 | $256.80 |
| + Startup overhead (~45 min) | +0.75 | +$16.47 | +$8.03 |

**Startup overhead for 72B:** instance boot (~2-3 min), model download (~30-40 min for ~144GB at FP16, or ~37GB for INT4-AWQ), vLLM model loading + tensor parallel init (~10-15 min). Total: ~45-60 minutes.

Using g5.48xlarge (cheaper but degraded performance):

| Scenario | Hours | On-Demand Cost | Spot Cost |
|----------|-------|---------------|-----------|
| 12-hour experiment | 12 | $195.46 | $82.57 |
| 24-hour experiment | 24 | $390.91 | $165.14 |

### Monthly Cost (Qwen2.5-VL-72B)

**p4d.24xlarge:**

| Experiments/Month | On-Demand (24hr each) | Spot (24hr each) |
|------------------|-----------------------|------------------|
| 1 | $543.51 | $264.83 |
| 2 | $1,087.02 | $529.66 |

**g5.48xlarge:**

| Experiments/Month | On-Demand (24hr each) | Spot (24hr each) |
|------------------|-----------------------|------------------|
| 1 | $403.12 | $170.27 |
| 2 | $806.25 | $340.55 |

**COST CRITICAL:** A single 24-hour experiment on the cheapest viable instance (g5.48xlarge spot) costs $165 -- 65% over the entire $100/month budget. The p4d.24xlarge on spot costs $257 per experiment -- 2.6x the monthly budget. This model is completely out of scope.

---

## Break-Even Analysis

The break-even question only makes sense for gpt-oss-20b, since Qwen2.5-VL-72B never breaks even at any realistic experiment volume under $100/month.

### gpt-oss-20b vs. OpenRouter Qwen3-8B

This is not a direct comparison because they are **different models**. But if the question is "at what point does self-hosting gpt-oss-20b become cheaper per-experiment than OpenRouter Qwen3-8B":

| OpenRouter Cost/Experiment | Self-Host Spot Cost/Experiment (g5.2xlarge, 24hr) | Break-Even |
|---------------------------|---------------------------------------------------|-----------|
| $0.50 | $11.41 | **Never** (self-host is 22.8x more expensive) |
| $2.50 | $11.41 | **Never** (self-host is 4.6x more expensive) |
| $5.00 | $11.41 | **Never** (self-host is 2.3x more expensive) |

Self-hosting gpt-oss-20b on AWS GPU instances is more expensive than OpenRouter API access at every experiment volume within the $100/month budget.

### What If Experiment Volume Were Much Higher?

Self-hosting amortizes fixed costs (startup time, model download) across longer continuous usage. If you ran experiments **continuously** (730 hours/month):

- g5.2xlarge spot, 730 hrs: $340/month
- That buys you dedicated, always-on inference for ~$340/month
- At 30 experiments/month (24hr each), that is ~$11.33/experiment -- still more than OpenRouter

The only scenario where self-hosting wins: **if OpenRouter costs exceed ~$11/experiment AND you need to run gpt-oss-20b specifically** (not available on OpenRouter at those rates).

---

## Budget Reality Check

| Scenario | Cost | % of $100 Budget | Verdict |
|----------|------|-----------------|---------|
| OpenRouter, 4 experiments | $2-$20 | 2-20% | Well within budget |
| gpt-oss-20b, g5.2xlarge spot, 1x 24hr | $11.41 | 11.4% | Fits, but expensive for one run |
| gpt-oss-20b, g5.2xlarge spot, 4x 24hr | $45.64 | 45.6% | Fits, leaves little margin |
| gpt-oss-20b, g5.2xlarge on-demand, 4x 24hr | $118.76 | **118.8%** | Over budget |
| Qwen2.5-VL-72B, g5.48xlarge spot, 1x 24hr | $170.27 | **170.3%** | Over budget on a single run |
| Qwen2.5-VL-72B, p4d.24xlarge spot, 1x 24hr | $264.83 | **264.8%** | Nearly 3x budget on one run |
| Qwen2.5-VL-72B, p4d.24xlarge on-demand, 1x 24hr | $543.51 | **543.5%** | 5.4x the entire budget |

---

## Throughput Estimates

| Model | Instance | Quantization | Est. Tokens/sec | Notes |
|-------|----------|-------------|-----------------|-------|
| gpt-oss-20b | g5.2xlarge (1x A10G) | MXFP4 | ~500-1,500 | MoE helps; only 3.6B active params |
| gpt-oss-20b | g5.12xlarge (4x A10G) | FP16 | ~2,000-4,000 | Overkill for this model |
| gpt-oss-20b | 1x A100 (reference) | FP16 | ~9,743 | Benchmark from GPUStack |
| Qwen2.5-VL-72B | g5.48xlarge (8x A10G) | INT4 | ~200-500 | No NVLink, poor TP scaling |
| Qwen2.5-VL-72B | p4d.24xlarge (8x A100) | INT8 | ~800-1,500 | NVLink helps, but 72B is slow |

For reference, OpenRouter's Qwen3-8B scales horizontally and effectively has no single-user throughput ceiling. A self-hosted single-GPU setup will be the bottleneck in your experiment pipeline.

---

## Startup Time Estimates

| Model | Instance | Steps | Est. Total Time |
|-------|----------|-------|----------------|
| gpt-oss-20b | g5.2xlarge | Boot (2min) + download 41GB (10-15min) + vLLM load (5-10min) | **17-27 min** |
| gpt-oss-20b | g5.2xlarge (cached on EBS) | Boot (2min) + vLLM load (5-10min) | **7-12 min** |
| Qwen2.5-VL-72B | p4d.24xlarge | Boot (3min) + download 144GB (20-40min) + vLLM TP load (10-15min) | **33-58 min** |
| Qwen2.5-VL-72B | p4d.24xlarge (cached on EBS) | Boot (3min) + vLLM TP load (10-15min) | **13-18 min** |

**Note:** Caching models on an EBS snapshot saves download time but adds ~$4/month per 100GB of gp3 storage. Worth it if running multiple experiments.

---

## Serving Architecture (If You Proceeded Anyway)

### gpt-oss-20b on g5.2xlarge

```bash
# 1. Launch g5.2xlarge (spot) with Deep Learning AMI
# 2. Download model
huggingface-cli download openai/gpt-oss-20b --local-dir /opt/models/gpt-oss-20b

# 3. Serve with vLLM (MXFP4 quantization)
python -m vllm.entrypoints.openai.api_server \
  --model /opt/models/gpt-oss-20b \
  --quantization mxfp4 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 16384 \
  --host 0.0.0.0 --port 8000

# 4. Point GEPA runner at local endpoint instead of OpenRouter
# Change model config to: http://localhost:8000/v1
```

### Qwen2.5-VL-72B on p4d.24xlarge

```bash
# 1. Launch p4d.24xlarge (spot) with Deep Learning AMI
# 2. Download model
huggingface-cli download Qwen/Qwen2.5-VL-72B-Instruct --local-dir /opt/models/qwen2.5-vl-72b

# 3. Serve with vLLM (tensor parallel across 8 A100s)
python -m vllm.entrypoints.openai.api_server \
  --model /opt/models/qwen2.5-vl-72b \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 16384 \
  --host 0.0.0.0 --port 8000
```

---

## What Would Need to Change for Self-Hosting to Make Sense

1. **Budget increase to $300-500/month** -- This opens up 2-4 experiments/month with gpt-oss-20b on spot instances, or 1-2 with the 72B model on g5.48xlarge spot.

2. **Experiment volume increase to 20+/month** -- At very high volume, a reserved instance or persistent spot cluster amortizes the fixed per-startup costs and begins to undercut per-token API pricing.

3. **Need for a specific model not available on OpenRouter** -- If gpt-oss-20b or Qwen2.5-VL-72B specifically is required (not just any 20B/72B model), self-hosting may be the only option regardless of cost.

4. **Latency/privacy requirements** -- If data cannot leave your VPC, self-hosting is mandatory. This is a compliance decision, not a cost decision.

5. **Switch to smaller quantized models** -- If a 7-8B model suffices, self-hosting on a g5.xlarge spot (~$0.39/hr) becomes very cheap: $4.68-$9.36 per 12-24hr experiment. But at that point, OpenRouter's Qwen3-8B at $0.50-$5.00/experiment is still cheaper with zero operational burden.

---

## Final Verdict

| Question | Answer |
|----------|--------|
| Is self-hosting gpt-oss-20b viable under $100/mo? | Technically yes (spot), but it costs more per experiment than OpenRouter |
| Is self-hosting Qwen2.5-VL-72B viable under $100/mo? | **No.** A single experiment exceeds the entire monthly budget |
| Is self-hosting cheaper than OpenRouter at current scale? | **No.** Not even close |
| What is the minimum budget for self-hosting to make sense? | ~$300/mo for gpt-oss-20b, ~$500/mo for the 72B model |
| What is the minimum experiment volume? | 20+/month continuous usage for cost parity |

**Stick with OpenRouter.** The $0.50-$5.00 per experiment cost with zero infrastructure overhead, zero startup latency, and zero spot interruption risk is unbeatable at this scale. Self-hosting is a solution for a problem this project does not have.

---

## Sources

- [g5.2xlarge pricing - Vantage](https://instances.vantage.sh/aws/ec2/g5.2xlarge)
- [g5.4xlarge pricing - Vantage](https://instances.vantage.sh/aws/ec2/g5.4xlarge)
- [g5.12xlarge pricing - Vantage](https://instances.vantage.sh/aws/ec2/g5.12xlarge)
- [g5.48xlarge pricing - Vantage](https://instances.vantage.sh/aws/ec2/g5.48xlarge)
- [p3.2xlarge pricing - Vantage](https://instances.vantage.sh/aws/ec2/p3.2xlarge)
- [p3.8xlarge pricing - Vantage](https://instances.vantage.sh/aws/ec2/p3.8xlarge)
- [p4d.24xlarge pricing - Vantage](https://instances.vantage.sh/aws/ec2/p4d.24xlarge)
- [EC2 On-Demand Pricing - AWS](https://aws.amazon.com/ec2/pricing/on-demand/)
- [EC2 Spot Pricing - AWS](https://aws.amazon.com/ec2/spot/pricing/)
- [AWS GPU Instance Pricing Comparison - DoiT](https://compute.doit.com/gpu)
- [gpt-oss-20b - Hugging Face](https://huggingface.co/openai/gpt-oss-20b)
- [Qwen2.5-VL-72B VRAM Needs - Novita AI](https://blogs.novita.ai/qwen2-5-vl-72b-vram/)
- [vLLM GPU Benchmark A100 - DatabaseMart](https://www.databasemart.com/blog/vllm-gpu-benchmark-a100-40gb)
- [GPT-OSS-20B Throughput on A100 - GPUStack](https://docs.gpustack.ai/2.0/performance-lab/gpt-oss-20b/a100/)
- [Qwen2.5-VL vLLM Usage Guide](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen2.5-VL.html)
- [AWS EC2 GPU Pricing Guide - TRG Datacenters](https://www.trgdatacenters.com/resource/aws-gpu-pricing/)
- [GPU Spot Instance Interruption Rates - ThunderCompute](https://www.thundercompute.com/blog/should-i-use-cloud-gpu-spot-instances)
- [AWS GPU Pricing Update 2025 - Pump](https://www.pump.co/blog/aws-ec2-pricing-update)
