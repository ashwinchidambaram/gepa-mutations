# RayCluster Model Integration: Synthesis & Decision Matrix

**Date:** 2026-03-22
**Budget:** $100/month hard ceiling
**Current setup:** OpenRouter API → Qwen3-8B ($0.50-$5.00/experiment)

---

## Executive Summary

Three viable paths exist, each with distinct tradeoffs:

1. **RayCluster via Tailscale ($0/month)** — free access to gpt-oss-20b and Qwen2.5-VL-72B, but results are NOT comparable to paper baselines (different models).
2. **OpenRouter ($0.50-5.00/experiment)** — current setup, paper-comparable Qwen3-8B, zero infrastructure.
3. **Self-host Qwen3-8B ($5-10/experiment on AWS/GCP spot)** — same model as paper for exact reproducibility, fits within $100/month budget at 2-4 experiments, but adds operational overhead.

Self-hosting the RayCluster models (gpt-oss-20b, 72B) on cloud GPUs is NOT recommended — the 72B model exceeds the budget on a single experiment, and the 20B model offers no advantage over accessing it free via the RayCluster.

---

## Decision Matrix: All Options Compared

### Network Connectivity (How to reach 10.0.10.66:8123)

| Option | Monthly Cost | Setup Time | Reliability | SV Admin Burden | Cloud | Verdict |
|--------|-------------|------------|-------------|-----------------|-------|---------|
| **Tailscale (subnet router)** | **$0** | **15-30 min** | **Excellent** | **Minimal** | **Both** | **STRONG YES** |
| Self-hosted WireGuard | $0-5 | 30-60 min | High | Low-Moderate | Both | YES (backup) |
| AWS Site-to-Site VPN | $36.50 | 1-2 days | Excellent | Moderate (IPsec) | AWS | YES (if Tailscale blocked) |
| GCP Cloud VPN (single) | $36.50 | 2-4 hrs | High | Moderate (IPsec) | GCP | MAYBE (budget pressure) |
| GCP Cloud VPN (HA, 2 tunnels) | $73.00 | 2-4 hrs | Very High | Moderate | GCP | MAYBE (tight budget) |
| Reverse SSH tunnel | $0 | 15 min | Medium | None | Both | MAYBE (dev only) |
| SSH via user laptop | $0 | 10 min | Low | None | Both | MAYBE (fragile) |
| Cloudflare Tunnel | $0 | 30-60 min | High | Low | Both | MAYBE (security concern) |
| AWS Client VPN | ~$87 | 2-3 hrs | N/A | N/A | AWS | NO (wrong tool) |
| AWS Direct Connect | $270+ | Weeks | Excellent | Heavy | AWS | NO (overkill) |
| GCP Cloud Interconnect | $245-1,754+ | Weeks | Very High | Heavy | GCP | NO (overkill) |

### Self-Hosting Qwen3-8B (Paper Model) on Cloud GPUs

| Option | Cost/Experiment (spot, 24hr) | Monthly (4 exp) | % of Budget | Throughput | Verdict |
|--------|------------------------------|-----------------|-------------|------------|---------|
| **OpenRouter Qwen3-8B** | **$0.50-5.00** | **$2-20** | **2-20%** | **Unlimited (API)** | **Cheapest, no infra** |
| **Qwen3-8B on AWS g6.xlarge (L4)** | **$9.82** | **$39.26** | **39%** | **~53 tok/s** | **Best self-host option** |
| Qwen3-8B on AWS g5.xlarge (A10G) | $9.91 | $39.65 | 40% | ~30-45 tok/s | Similar to L4, project already uses AWS |
| Qwen3-8B on GCP g2-standard-4 (L4) | $6.72 | $26.88 | 27% | ~30-80 tok/s | Cheapest per-hour, but less throughput |
| Qwen3-8B on GCP n1+T4 | $3.36-4.08 | $13-16 | 13-16% | ~3.8 tok/s (BF16) | **DISQUALIFIED** — too slow |
| Qwen3-8B on AWS g4dn.xlarge (T4) | $5.16 | $20.64 | 21% | ~3.8 tok/s (BF16) | **DISQUALIFIED** — too slow |

**Key finding:** Self-hosting Qwen3-8B is **within budget** on both clouds ($27-40/month at 4 experiments). However, at current experiment volumes (2-4/month), OpenRouter is still cheaper per-experiment. The argument for self-hosting is **scientific reproducibility** (pinned weights, controlled inference), not cost savings.

**Break-even vs OpenRouter:** ~25M output tokens/experiment at OpenRouter's $0.40/M output rate.

### Self-Hosting RayCluster Models (for reference)

| Option | Cost/Experiment (spot, 24hr) | Monthly (4 exp) | Verdict |
|--------|------------------------------|-----------------|---------|
| gpt-oss-20b on GCP g2-standard-4 (L4) | $6.76 | $27 | Fits budget, but use RayCluster for free instead |
| gpt-oss-20b on AWS g5.xlarge (A10G) | $9.91 | $40 | Fits budget, but use RayCluster for free instead |
| Qwen2.5-VL-72B (any cloud) | $87-257+ | $165-1,027+ | **NOT VIABLE** — exceeds budget on a single run |

---

## Recommendation Tiers

### Tier 1A: RayCluster via Tailscale (New models, $0)

**Use the SupportVectors RayCluster via Tailscale**

- **Monthly cost:** $0 (networking) + $0 (compute — models already running)
- **Setup:** Install Tailscale on one SV network machine as subnet router, add 3 lines to EC2/GCE startup script (15 min)
- **Models:** gpt-oss-20b (text), Qwen2.5-VL-72B (vision+text)
- **API:** OpenAI-compatible at `http://10.0.10.66:8123/v1`
- **Caveat:** Results NOT comparable to paper baselines (different models)

**Best for:** Exploring new model capabilities beyond the paper's scope.

### Tier 1B: OpenRouter (Paper model, status quo)

**Continue with OpenRouter API**

- **Monthly cost:** $2-20 (at 2-4 experiments)
- **Setup:** Already done
- **Model:** Qwen3-8B (exact paper model)
- **Caveat:** No control over model weights, inference quantization, or rate limits

**Best for:** Continuing paper reproduction and mutation experiments with minimal friction.

### Tier 2: Self-Host Qwen3-8B (Paper model, full control)

**Self-host Qwen3-8B on AWS g6.xlarge spot (L4 GPU)**

- **Monthly cost:** $39-40 at 4 experiments ($10/experiment)
- **Setup:** 2-4 hours (vLLM + model download + startup script)
- **Model:** Qwen3-8B with pinned weights and controlled inference parameters
- **Throughput:** ~53 tok/s (adequate for 12-24hr experiments)
- **Key advantage:** Exact scientific reproducibility — same model, pinned checkpoint, no API rate limits, controlled quantization

**Best for:** When reproducibility is a hard scientific requirement, or if OpenRouter changes Qwen3-8B weights/quantization without notice.

GCP alternative: g2-standard-4 spot at $6.72/experiment ($27/month) — 28% cheaper per-hour but the project already uses AWS.

### Tier 3: Self-Host RayCluster Models (Only if RayCluster is down)

Only if the RayCluster becomes unavailable AND you need gpt-oss-20b specifically:
- AWS g5.xlarge spot: $9.91/experiment, $40/month
- GCP g2-standard-4 spot: $6.76/experiment, $27/month

**Qwen2.5-VL-72B self-hosting is infeasible at any tier.** A single experiment ($87-257) exceeds the monthly budget.

---

## Key Uncertainties to Resolve

| Uncertainty | Impact | How to Resolve |
|-------------|--------|---------------|
| **Can you install Tailscale on an SV network machine?** | Blocks Tier 1 entirely | Ask SV admin or install yourself (you have SSH access) |
| **Is the RayCluster available 24/7?** | Affects experiment scheduling | Test uptime over a week; ask SV about maintenance windows |
| **What are the rate limits on the RayCluster API?** | Affects experiment throughput | Benchmark with a test workload (100 sequential API calls) |
| **Does gpt-oss-20b produce comparable quality to Qwen3-8B on GEPA benchmarks?** | Affects whether results are useful | Run one pilot experiment on HotpotQA and compare |
| **Does Qwen2.5-VL-72B work with text-only GEPA tasks?** | Determines if the VL model is usable | Test with a simple completion call |
| **SV RayCluster auth model** | The dummy key `"sv-openai-api-key"` suggests no real auth — could change | Confirm with SV admin that this access pattern is sanctioned |

---

## Code Change Summary

To support RayCluster (or any non-OpenRouter model endpoint), these files need modifications:

### 1. `src/gepa_mutations/config.py` — Add API base URL field

```python
# Add to Settings class:
api_base_url: str = ""  # Empty = use OpenRouter default; set to "http://10.0.10.66:8123/v1" for RayCluster
model_prefix: str = "openrouter"  # "openrouter" for OpenRouter, "openai" for OpenAI-compatible endpoints
```

### 2. `src/gepa_mutations/base.py` — Make model prefix configurable

Currently hardcodes `f"openrouter/{settings.gepa_model}"` in 3 places:
- `build_reflection_lm()` (line 76)
- `build_task_lm()` (line 86-87)
- `build_qa_task_lm()` (line 97-98)
- `config_snapshot()` (line 237)

**Change to:**
```python
def _model_id(settings: Settings) -> str:
    """Build the full model ID from settings."""
    if settings.model_prefix:
        return f"{settings.model_prefix}/{settings.gepa_model}"
    return settings.gepa_model
```

Then replace all `f"openrouter/{settings.gepa_model}"` with `_model_id(settings)`.

### 3. `src/gepa_mutations/runner/experiment.py` — Same pattern

The `ExperimentRunner` class has `_build_task_lm()`, `_build_reflection_lm()`, `_build_qa_task_lm()` with the same hardcoded prefix. Apply the same `_model_id()` helper.

If using LiteLLM (which GEPA uses internally), also pass `api_base` when `settings.api_base_url` is set:
```python
# In LM() constructor calls, add:
api_base=settings.api_base_url or None
```

LiteLLM's `completion()` already supports the `api_base` parameter, so this is the cleanest integration point.

### 4. `.env` — Configuration for RayCluster

```bash
# For RayCluster:
GEPA_MODEL=openai/gpt-oss-20b
API_BASE_URL=http://10.0.10.66:8123/v1
MODEL_PREFIX=openai
OPENROUTER_API_KEY=sv-openai-api-key  # dummy key for RayCluster

# For OpenRouter (current):
GEPA_MODEL=qwen/qwen3-8b
API_BASE_URL=
MODEL_PREFIX=openrouter
OPENROUTER_API_KEY=<real key>
```

### Estimated effort: ~30 minutes of code changes, mostly in `config.py` and `base.py`.

---

## Cost Summary Table

| Path | Monthly Cost | Per-Experiment | Setup Time | Model Comparability | Risk |
|------|-------------|----------------|------------|---------------------|------|
| **Tier 1A: RayCluster + Tailscale** | **$0** | **$0** | **15 min** | **No (different models)** | **SV admin cooperation** |
| **Tier 1B: OpenRouter (status quo)** | **$2-20** | **$0.50-5.00** | **Done** | **Yes (Qwen3-8B)** | **None** |
| **Tier 2: Self-host Qwen3-8B (AWS L4)** | **$39-40** | **~$10** | **2-4 hrs** | **Yes (pinned weights)** | **Ops overhead** |
| Tier 2 alt: Self-host Qwen3-8B (GCP L4) | $27 | $6.72 | 2-4 hrs | Yes (pinned weights) | Ops overhead, new cloud |
| Tier 3: Self-host gpt-oss-20b (AWS/GCP) | $27-40 | $6.76-9.91 | 2-4 hrs | No | Use RayCluster instead |
| NOT VIABLE: Self-host 72B (any cloud) | $165-1,027+ | $87-257+ | 4-8 hrs | No | Over budget |

---

## Sources

All pricing verified via web search on 2026-03-22. Detailed breakdowns with per-instance pricing, throughput estimates, and break-even analyses are in:
- `docs/aws-qwen3-8b-self-hosting.md` (AWS — Qwen3-8B, **revised**)
- `docs/gcp-qwen3-8b-self-hosting.md` (GCP — Qwen3-8B, **revised**)
- `docs/self-hosting-cost-analysis.md` (AWS — gpt-oss-20b/72B, original analysis)
- `docs/gcp-self-hosting-cost-analysis.md` (GCP — gpt-oss-20b/72B, original analysis)
- Task 2 output (AWS → RayCluster connectivity)
- Task 5 output (GCP → RayCluster connectivity)
