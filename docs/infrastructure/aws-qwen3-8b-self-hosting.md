# Self-Hosting Qwen3-8B on AWS: Cost Analysis

**Date:** 2026-03-22
**Model:** Qwen/Qwen3-8B (8B dense parameters)
**Budget Ceiling:** $100/month hard limit
**Current approach:** OpenRouter API at $0.05/$0.40 per M tokens (input/output)
**Current cost per experiment:** $0.50-$5.00
**Experiment runtime:** 12-24 hours each
**Experiment volume:** 2-4 experiments/month

---

## Executive Summary

**Recommendation: Self-hosting Qwen3-8B on a g6.xlarge (L4) or g5.xlarge (A10G) spot instance is viable and potentially cost-effective at higher experiment volumes, but at current scale (2-4 experiments/month), the savings are marginal and the operational overhead is not worth it. Stick with OpenRouter unless experiment volume increases to 6+ per month.**

Unlike the previous analysis of gpt-oss-20b (which required expensive multi-GPU setups), Qwen3-8B is a small 8B dense model that fits comfortably on any modern GPU instance. The economics are fundamentally different: self-hosting an 8B model costs $4.91-$9.81 per 12-24hr experiment on a g6.xlarge spot -- comparable to the upper range of OpenRouter costs. The critical advantage is that self-hosting the SAME model (Qwen3-8B) means results remain directly comparable to paper baselines. There is no model comparability issue.

---

## Model: Qwen3-8B VRAM Requirements

| Precision | VRAM Required | Fits On |
|-----------|--------------|---------|
| BF16/FP16 | ~16 GB | A10G (24GB), L4 (24GB), T4 (16GB -- very tight) |
| FP8 | ~9 GB | A10G, L4, T4 |
| INT8 | ~8 GB | A10G, L4, T4 |
| INT4 (AWQ/GPTQ) | ~5-6 GB | Everything, including consumer GPUs |

Source: [Qwen3-8B VRAM specs (apxml.com)](https://apxml.com/models/qwen3-8b), [Qwen official speed benchmarks](https://qwen.readthedocs.io/en/latest/getting_started/speed_benchmark.html)

**Key insight:** At BF16, the model alone is ~16GB. On a T4 (16GB VRAM), there is zero headroom for KV cache, which means the T4 can only run Qwen3-8B with quantization (INT8 or INT4). On A10G (24GB) or L4 (24GB), the model fits at full BF16 with ~8GB remaining for KV cache -- plenty for single-user inference with moderate context lengths.

---

## Instance Recommendations

| Instance | GPU | VRAM | vCPUs | RAM | On-Demand ($/hr) | Spot ($/hr) | Spot Savings | Can Run Qwen3-8B? |
|----------|-----|------|-------|-----|-------------------|-------------|-------------|-------------------|
| **g6.xlarge** | 1x NVIDIA L4 | 24 GB | 4 | 16 GiB | $0.805 | **$0.409** | 49% | Yes -- BF16 with KV cache headroom |
| **g5.xlarge** | 1x NVIDIA A10G | 24 GB | 4 | 16 GiB | $1.006 | **$0.413** | 59% | Yes -- BF16 with KV cache headroom |
| **g4dn.xlarge** | 1x NVIDIA T4 | 16 GB | 4 | 16 GiB | $0.526 | **$0.215** | 59% | Yes, but INT8/INT4 only -- tight fit |
| **inf2.xlarge** | 1x Inferentia2 | 32 GB | 4 | 16 GiB | $0.758 | **$0.136** | 82% | Possible but with caveats (see below) |

Sources: [Vantage g6.xlarge](https://instances.vantage.sh/aws/ec2/g6.xlarge), [Vantage g5.xlarge](https://instances.vantage.sh/aws/ec2/g5.xlarge), [Vantage g4dn.xlarge](https://instances.vantage.sh/aws/ec2/g4dn.xlarge), [Vantage inf2.xlarge](https://instances.vantage.sh/aws/ec2/inf2.xlarge). All prices us-east-1, verified 2026-03-22.

### Instance-by-Instance Assessment

**g6.xlarge (L4) -- RECOMMENDED if self-hosting:**
- L4 is Ada Lovelace architecture (newer than A10G's Ampere)
- 24GB VRAM fits BF16 comfortably with KV cache headroom
- Spot price ($0.409/hr) is nearly identical to g5.xlarge spot but on-demand is cheaper
- L4 has excellent FP8 support for further optimization
- Qwen 7B-class models benchmarked at **~53 tokens/sec** on L4 (single batch, BF16)

**g5.xlarge (A10G) -- STRONG ALTERNATIVE:**
- A10G is well-established for LLM inference on AWS
- 24GB VRAM, same headroom as L4
- Spot price ($0.413/hr) virtually identical to g6.xlarge
- A10G benchmarked at **~30-45 tokens/sec** for 7-8B models (single batch)
- Better ecosystem support -- more vLLM testing has been done on A10G than L4

**g4dn.xlarge (T4) -- NOT RECOMMENDED:**
- T4 is Turing architecture (2018), no BF16 support, limited FP16 throughput
- 16GB VRAM means Qwen3-8B at BF16 does NOT fit with any KV cache -- must quantize to INT8
- Benchmarked at **~3.8 tokens/sec** for Qwen 7B-class models on T4. This is catastrophically slow.
- At 3.8 tok/s, generating a 500-token response takes ~132 seconds. Over thousands of API calls, this adds HOURS to experiment runtime.
- The spot price ($0.215/hr) looks cheap, but the throughput penalty means experiments take 5-10x longer, negating the savings.

**inf2.xlarge (Inferentia2) -- NOT RECOMMENDED (for now):**
- Attractive spot price ($0.136/hr) and 32GB accelerator memory
- **Compatibility problem:** Qwen3 models with `tie_word_embeddings=true` encounter errors in vLLM's NxD Inference on Neuron. The Qwen team has documented workarounds for Qwen2.5 on Inferentia2 via HuggingFace TGI, but Qwen3-8B support on Inferentia2 is not yet mature.
- Requires model compilation for Neuron, which does not support dynamic input shapes -- needs fixed sequence lengths
- If you are willing to spend engineering time on Neuron compilation and debugging, this could be the cheapest option. But it is not plug-and-play like GPU instances with vLLM.

Sources: [Qwen Inferentia2 guide](https://aws.amazon.com/blogs/machine-learning/how-to-run-qwen-2-5-on-aws-ai-chips-using-hugging-face-libraries/), [vLLM Neuron docs](https://docs.vllm.ai/en/v0.10.1/getting_started/installation/aws_neuron.html), [vLLM NxD Inference user guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/developer_guides/vllm-user-guide-v1.html)

---

## Throughput Estimates

### Official Qwen3-8B Benchmarks (SGLang on H20 96GB -- reference ceiling)

| Context Length | BF16 (tok/s) | FP8 (tok/s) | AWQ-INT4 (tok/s) |
|---------------|-------------|------------|-----------------|
| 1 | 82 | 150 | 144 |
| 6,144 | 296 | 517 | 478 |
| 14,336 | 525 | 860 | 770 |
| 30,720 | 833 | 1,242 | 1,076 |

Source: [Qwen3 Speed Benchmark](https://qwen.readthedocs.io/en/latest/getting_started/speed_benchmark.html)

These are H20 numbers (reference ceiling). Actual throughput on cheaper GPUs will be lower.

### Estimated Throughput on AWS Instance GPUs

| GPU | Architecture | Qwen3-8B (BF16) est. tok/s | Qwen3-8B (INT8/INT4) est. tok/s | Notes |
|-----|-------------|---------------------------|--------------------------------|-------|
| L4 (g6.xlarge) | Ada Lovelace | ~50-60 | ~80-120 | Based on Qwen 7B benchmark: 53 tok/s on L4 |
| A10G (g5.xlarge) | Ampere | ~30-45 | ~50-80 | Based on Llama-7B on A10: 44 tok/s; similar architecture |
| T4 (g4dn.xlarge) | Turing | ~3-5 | ~15-25 | Based on Qwen 7B benchmark: 3.8 tok/s on T4 at BF16. INT8 helps significantly. |
| H20 (reference) | Hopper | 82 | 150 | Official Qwen benchmark |

Sources: [Qwen GPU benchmarks (Medium)](https://medium.com/@wltsankalpa/benchmarking-qwen-models-across-nvidia-gpus-t4-l4-h100-architectures-finding-your-sweet-spot-a59a0adf9043), [Koyeb GPU benchmarks](https://www.koyeb.com/docs/hardware/gpu-benchmarks), [Nucleusbox GPU comparison](https://www.nucleusbox.com/choose-gpu-for-llms-t4-a10-a100/)

**Critical finding:** The L4 is roughly 10-15x faster than the T4 for this workload. The T4's 3.8 tok/s at BF16 is disqualifying for experiment workloads that make thousands of inference calls. Even with INT8 quantization improving the T4 to ~15-25 tok/s, it is still 2-4x slower than the L4 or A10G at BF16.

### Throughput Impact on Experiment Runtime

GEPA experiments make thousands of API calls per run. If a typical experiment generates ~2M output tokens total:

| GPU | Est. tok/s | Time to generate 2M tokens | Impact vs. OpenRouter |
|-----|-----------|---------------------------|----------------------|
| L4 (BF16) | ~55 | ~10.1 hours | Comparable -- OpenRouter is not throughput-limited |
| A10G (BF16) | ~40 | ~13.9 hours | Slightly slower |
| T4 (INT8) | ~20 | ~27.8 hours | Doubles experiment time |
| T4 (BF16) | ~4 | ~138.9 hours | 5.8 DAYS -- completely unacceptable |
| OpenRouter | Elastic | N/A (rate-limited, not throughput-limited) | Baseline |

**The T4 at BF16 would turn a 12-hour experiment into a 6-day experiment. The T4 is eliminated.**

---

## Cost Per Experiment

### Using g6.xlarge (L4) -- Recommended Self-Host Option

| Scenario | Hours | On-Demand Cost | Spot Cost |
|----------|-------|---------------|-----------|
| 12-hour experiment | 12 | $9.66 | **$4.91** |
| 24-hour experiment | 24 | $19.32 | **$9.81** |
| + Startup overhead (~15 min) | +0.25 | +$0.20 | +$0.10 |

### Using g5.xlarge (A10G) -- Alternative

| Scenario | Hours | On-Demand Cost | Spot Cost |
|----------|-------|---------------|-----------|
| 12-hour experiment | 12 | $12.07 | **$4.96** |
| 24-hour experiment | 24 | $24.14 | **$9.91** |
| + Startup overhead (~15 min) | +0.25 | +$0.25 | +$0.10 |

### Using g4dn.xlarge (T4) -- NOT RECOMMENDED (runtime penalty)

| Scenario | Hours | On-Demand Cost | Spot Cost |
|----------|-------|---------------|-----------|
| 12-hour experiment (but takes 28hr) | 28 | $14.73 | **$6.02** |
| 24-hour experiment (but takes 56hr) | 56 | $29.46 | **$12.04** |

The T4's low hourly rate is a trap. The throughput penalty means experiments run 2-3x longer (INT8) or 6x longer (BF16), so total cost per experiment is higher than the L4/A10G despite the lower hourly rate.

### Startup Overhead

| Step | Time Estimate | Notes |
|------|--------------|-------|
| Instance boot | ~2 min | Standard EC2 boot |
| Model download (~16GB BF16) | ~5-8 min | HuggingFace Hub over EC2 network |
| vLLM server startup + model load | ~3-5 min | Single GPU, no tensor parallelism |
| **Total** | **~10-15 min** | Fast -- this is a small model |

Caching the model on an EBS snapshot saves 5-8 minutes of download time. At ~16GB, the gp3 storage costs ~$1.28/month -- worth it if running multiple experiments.

---

## Monthly Cost Projection

### Self-Hosted on g6.xlarge (L4) Spot

| Experiments/Month | 12hr each | 24hr each |
|------------------|-----------|-----------|
| 2 | **$10.02** | **$19.82** |
| 3 | **$15.03** | **$29.73** |
| 4 | **$20.04** | **$39.64** |
| 6 | **$30.06** | **$59.46** |
| 8 | **$40.08** | **$79.28** |

### Self-Hosted on g5.xlarge (A10G) Spot

| Experiments/Month | 12hr each | 24hr each |
|------------------|-----------|-----------|
| 2 | **$10.12** | **$20.02** |
| 3 | **$15.18** | **$30.03** |
| 4 | **$20.24** | **$40.04** |
| 6 | **$30.36** | **$60.06** |
| 8 | **$40.48** | **$80.08** |

### OpenRouter API (Current)

| Experiments/Month | Low end ($0.50 each) | High end ($5.00 each) |
|------------------|---------------------|----------------------|
| 2 | **$1.00** | **$10.00** |
| 3 | **$1.50** | **$15.00** |
| 4 | **$2.00** | **$20.00** |
| 6 | **$3.00** | **$30.00** |
| 8 | **$4.00** | **$40.00** |

All scenarios fit within the $100/month budget. The question is which provides better value.

---

## Break-Even Analysis: Self-Hosting vs. OpenRouter

The break-even depends heavily on the actual OpenRouter cost per experiment. The $0.50-$5.00 range is wide.

### At $0.50/experiment (low token usage)

| Metric | OpenRouter | g6.xlarge Spot (24hr) |
|--------|-----------|----------------------|
| Cost per experiment | $0.50 | $9.91 |
| Break-even | **Never** -- OpenRouter is 19.8x cheaper | |
| Monthly at 4 experiments | $2.00 | $39.64 |

Self-hosting makes no sense at this token volume. OpenRouter wins by a factor of 20.

### At $2.50/experiment (moderate token usage)

| Metric | OpenRouter | g6.xlarge Spot (24hr) |
|--------|-----------|----------------------|
| Cost per experiment | $2.50 | $9.91 |
| Break-even | **Never** -- OpenRouter is 4.0x cheaper | |
| Monthly at 4 experiments | $10.00 | $39.64 |

OpenRouter still wins comfortably.

### At $5.00/experiment (high token usage)

| Metric | OpenRouter | g6.xlarge Spot (24hr) |
|--------|-----------|----------------------|
| Cost per experiment | $5.00 | $9.91 |
| Break-even | **Never** -- OpenRouter is 2.0x cheaper | |
| Monthly at 4 experiments | $20.00 | $39.64 |

OpenRouter still wins.

### At $10.00/experiment (very high token usage -- possible with longer runs)

| Metric | OpenRouter | g6.xlarge Spot (24hr) |
|--------|-----------|----------------------|
| Cost per experiment | $10.00 | $9.91 |
| **Break-even** | **YES** -- nearly identical | |
| Monthly at 4 experiments | $40.00 | $39.64 |

Self-hosting breaks even only when OpenRouter experiments cost ~$10 each.

### At $10+/experiment (heavy token usage, many iterations)

Self-hosting on a g6.xlarge spot wins. The spot instance cost is fixed at ~$9.91/24hr regardless of how many tokens you generate. OpenRouter scales linearly with token count.

### Break-Even Token Volume

At OpenRouter's $0.40/M output tokens (the dominant cost):
- g6.xlarge spot for 24hr = $9.91
- $9.91 / $0.40 per M tokens = **~24.8M output tokens**

**If your experiment generates more than ~25M output tokens, self-hosting is cheaper.**

For a 12-hour experiment on g6.xlarge spot ($4.91):
- $4.91 / $0.40 per M tokens = **~12.3M output tokens**

**If your 12-hour experiment generates more than ~12M output tokens, self-hosting is cheaper.**

---

## Key Advantage: Model Comparability

**This is the strongest argument for self-hosting Qwen3-8B, independent of cost.**

With OpenRouter, you are already running Qwen3-8B -- the same model used in the GEPA paper. Self-hosting the same model means:

1. **Results remain directly comparable to paper baselines.** No model substitution, no asterisks.
2. **Full control over inference parameters.** You control temperature, top_p, top_k, max_tokens, and crucially, the exact model revision and quantization. OpenRouter may silently update the model version or route to different backends.
3. **No rate limiting.** OpenRouter rate limits can throttle experiments during peak hours. Self-hosted inference has consistent throughput.
4. **No API downtime risk.** A 24-hour experiment is vulnerable to OpenRouter outages. Self-hosted inference eliminates this dependency.
5. **Reproducibility.** Pin the exact model checkpoint, quantization, and vLLM version. OpenRouter's backend is a black box.

These advantages matter for a research project where reproducibility is paramount, even if the cost savings are marginal.

---

## Spot Interruption Risk

GPU spot instances on AWS typically have interruption rates in the **<5% to 10%** range for mainstream instance types. For a 24-hour experiment:

- At <5% interruption rate: ~1 in 20 experiments will be interrupted
- At 5-10% interruption rate: ~1 in 10 experiments will be interrupted

**Mitigation:**
- GEPA experiments already need checkpointing for fault tolerance
- Spot interruption gives a 2-minute warning -- enough to save state
- For a 24-hour experiment, the expected cost of one interruption + restart is ~$0.50-1.00 (startup overhead)
- Use multiple AZs in the spot request to maximize availability

Source: [AWS Spot Instance Advisor](https://aws.amazon.com/ec2/spot/instance-advisor/), [ThunderCompute spot analysis](https://www.thundercompute.com/blog/cloud-gpu-spot-instance-availability)

---

## Serving Architecture (If You Proceed)

```bash
#!/bin/bash
# Launch g6.xlarge (spot) with Deep Learning AMI
# User data script for experiment instance

set -euo pipefail

# Install dependencies
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Download model (or use cached EBS snapshot)
pip install huggingface-hub
huggingface-cli download Qwen/Qwen3-8B --local-dir /opt/models/qwen3-8b

# Serve with vLLM (BF16 on L4/A10G -- fits in 24GB)
pip install vllm
python -m vllm.entrypoints.openai.api_server \
  --model /opt/models/qwen3-8b \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 16384 \
  --host 127.0.0.1 --port 8000 &

# Wait for server to be ready
sleep 30

# Clone and setup experiment
git clone <repo> /opt/gepa-mutations
cd /opt/gepa-mutations
uv sync

# Point GEPA runner at local vLLM (OpenAI-compatible endpoint)
export OPENAI_API_BASE="http://127.0.0.1:8000/v1"
export OPENAI_API_KEY="not-needed"

# Run experiment
uv run python -m gepa_mutations.runner.run \
  --config s3://gepa-mutations-{id}/experiments/{exp-id}/config.json

# Upload results and self-terminate
aws s3 sync ./results/ s3://gepa-mutations-{id}/experiments/{exp-id}/results/
aws ec2 terminate-instances --instance-ids \
  $(TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" \
    -H "X-aws-ec2-metadata-token-ttl-seconds: 21600") && \
    curl -s -H "X-aws-ec2-metadata-token: $TOKEN" \
    http://169.254.169.254/latest/meta-data/instance-id)
```

### Integration with LiteLLM

GEPA uses LiteLLM internally. To redirect from OpenRouter to a local vLLM server, the model string would change from `openrouter/qwen/qwen3-8b` to `openai/Qwen/Qwen3-8B` with `api_base` set to `http://127.0.0.1:8000/v1`. This is a one-line config change.

---

## Budget Reality Check

| Scenario | Monthly Cost | % of $100 Budget | Verdict |
|----------|-------------|-----------------|---------|
| OpenRouter, 4 experiments at $2.50 each | $10.00 | 10% | Well within budget |
| OpenRouter, 4 experiments at $5.00 each | $20.00 | 20% | Well within budget |
| g6.xlarge spot, 4 x 12hr experiments | $20.04 | 20% | Well within budget |
| g6.xlarge spot, 4 x 24hr experiments | $39.64 | 40% | Within budget |
| g5.xlarge spot, 4 x 24hr experiments | $40.04 | 40% | Within budget |
| g6.xlarge on-demand, 4 x 24hr experiments | $78.28 | 78% | Tight but fits |
| g4dn.xlarge spot, 4 x 56hr (runtime-adjusted) | $48.16 | 48% | Fits but terrible value |

Every self-hosting option fits within the $100/month budget. The question is value, not viability.

---

## Decision Matrix

| Factor | OpenRouter | g6.xlarge Spot (L4) | g5.xlarge Spot (A10G) | g4dn.xlarge Spot (T4) |
|--------|-----------|--------------------|-----------------------|----------------------|
| Cost per 24hr experiment | $0.50-$5.00 | $9.91 | $10.01 | $12.04 (runtime-adj) |
| Cost per 12hr experiment | $0.25-$2.50 | $4.91 | $4.96 | $6.02 (runtime-adj) |
| Throughput (tok/s) | Elastic | ~55 | ~40 | ~20 (INT8) |
| Model comparability | Same model | Same model | Same model | Same model |
| Reproducibility | Low (black box backend) | High (pinned checkpoint) | High | High |
| Operational overhead | Zero | Moderate | Moderate | Moderate |
| Startup time | Instant | ~15 min | ~15 min | ~15 min |
| Rate limiting risk | Yes | No | No | No |
| Spot interruption risk | N/A | Yes (<5-10%) | Yes (<5-10%) | Yes (<5-10%) |
| Experiment runtime impact | None | None | +15-30% | +100-300% |

---

## Final Recommendation

### For current scale (2-4 experiments/month): Stick with OpenRouter.

At $0.50-$5.00 per experiment, OpenRouter is 2-20x cheaper than self-hosting with zero operational overhead. The total monthly spend ($2-$20) is a fraction of the budget. Self-hosting at this volume costs more and adds infrastructure complexity.

### For higher scale (6+ experiments/month) OR heavy token experiments ($10+/experiment): Consider g6.xlarge spot.

If experiment volume increases or individual experiments become token-heavy (>12M output tokens per run), self-hosting on a g6.xlarge (L4) spot instance becomes cost-competitive. The g6.xlarge spot at $0.409/hr provides excellent throughput (~55 tok/s) at a fixed hourly rate regardless of token volume.

### For reproducibility-critical work: Self-host regardless of cost.

If you need to guarantee exact model version, quantization, and inference parameters for publishable results, self-hosting provides control that OpenRouter cannot. The ~$5-10 premium per experiment is the cost of reproducibility. This is a research judgment call, not a cost optimization.

### Never use g4dn.xlarge (T4) for this workload.

The T4's throughput (3.8 tok/s at BF16, ~20 tok/s at INT8) is too slow for GEPA experiments. The low hourly rate is a trap -- experiments take 2-6x longer, costing more in total and blocking your experiment pipeline.

### Defer Inferentia2 (inf2.xlarge) until ecosystem matures.

The inf2.xlarge spot price ($0.136/hr) is extremely attractive, and its 32GB accelerator memory is more than sufficient. But Qwen3-8B compatibility issues with vLLM on Neuron make it a risky choice today. Revisit when AWS Neuron SDK has stable Qwen3 support.

---

## What Would Change This Recommendation

1. **OpenRouter pricing increases.** If Qwen3-8B pricing rises above $1.00/M output tokens, self-hosting becomes cheaper at current experiment volumes.
2. **Experiment volume increases to 10+/month.** At 10 experiments/month x 24hr, g6.xlarge spot costs $98/month -- still under budget but with unlimited token throughput.
3. **Token-heavy experiments.** If a new benchmark or mutation generates 25M+ output tokens per experiment, self-hosting wins on cost alone.
4. **Inferentia2 Qwen3 support matures.** An inf2.xlarge spot at $0.136/hr would cost $1.63/12hr or $3.26/24hr -- cheaper than OpenRouter at almost any experiment cost.
5. **Spot price changes.** GPU spot prices fluctuate. Monitor g6.xlarge spot in us-east-1; if it drops below $0.30/hr, self-hosting becomes compelling at lower volumes.

---

## Sources

- [g6.xlarge pricing - Vantage](https://instances.vantage.sh/aws/ec2/g6.xlarge)
- [g5.xlarge pricing - Vantage](https://instances.vantage.sh/aws/ec2/g5.xlarge)
- [g4dn.xlarge pricing - Vantage](https://instances.vantage.sh/aws/ec2/g4dn.xlarge)
- [inf2.xlarge pricing - Vantage](https://instances.vantage.sh/aws/ec2/inf2.xlarge)
- [Qwen3-8B VRAM requirements - apxml.com](https://apxml.com/models/qwen3-8b)
- [Qwen3 Speed Benchmark - Official](https://qwen.readthedocs.io/en/latest/getting_started/speed_benchmark.html)
- [Qwen GPU benchmarks (T4, L4, H100) - Medium](https://medium.com/@wltsankalpa/benchmarking-qwen-models-across-nvidia-gpus-t4-l4-h100-architectures-finding-your-sweet-spot-a59a0adf9043)
- [Koyeb GPU LLM Performance Benchmarks](https://www.koyeb.com/docs/hardware/gpu-benchmarks)
- [GPU comparison for LLMs - Nucleusbox](https://www.nucleusbox.com/choose-gpu-for-llms-t4-a10-a100/)
- [Qwen3-8B on OpenRouter](https://openrouter.ai/qwen/qwen3-8b)
- [Qwen on Inferentia2 - AWS Blog](https://aws.amazon.com/blogs/machine-learning/how-to-run-qwen-2-5-on-aws-ai-chips-using-hugging-face-libraries/)
- [vLLM Neuron installation](https://docs.vllm.ai/en/v0.10.1/getting_started/installation/aws_neuron.html)
- [AWS Spot Instance Advisor](https://aws.amazon.com/ec2/spot/instance-advisor/)
- [GPU Spot Instance Interruption Rates - ThunderCompute](https://www.thundercompute.com/blog/cloud-gpu-spot-instance-availability)
- [EC2 On-Demand Pricing - AWS](https://aws.amazon.com/ec2/pricing/on-demand/)
- [EC2 Spot Pricing - AWS](https://aws.amazon.com/ec2/spot/pricing/)
- [DoiT GPU Spot Price Comparison](https://compute.doit.com/gpu)
- [Llama 3.1 8B vLLM benchmarks - Microsoft](https://techcommunity.microsoft.com/blog/azurehighperformancecomputingblog/inference-performance-of-llama-3-1-8b-using-vllm-across-various-gpus-and-cpus/4448420)
- [Anyscale GPU selection guide](https://docs.anyscale.com/llm/serving/gpu-guidance)
