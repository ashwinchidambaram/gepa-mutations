# gepa-mutations

Experimental framework for reproducing and extending [GEPA](https://arxiv.org/abs/2507.19457) (ICLR 2026 Oral) — an automatic prompt evolution framework that uses a language model to iteratively improve task-specific prompts via natural-language reflection and Pareto-frontier selection.

## Background

GEPA replaces RL-based prompt optimization with an evolutionary loop:

1. **Trajectory Sampling** — collect execution traces on a minibatch of training examples
2. **Natural Language Reflection** — an LLM diagnoses failure patterns and proposes prompt improvements
3. **Pareto Frontier Selection** — maintain a diverse population, each candidate best on different examples
4. **System-Aware Merge** — crossover operator that combines complementary improvements

The paper reports +9.62pp aggregate improvement over baseline Qwen3-8B across 6 benchmarks, outperforming GRPO and MIPROv2 using up to 35× fewer rollouts.

This project reproduces the GEPA baseline and systematically tests 11 algorithm mutations across 4 model sizes (1.7B → 27B-AWQ).

---

## Methods

The 12 methods (GEPA baseline + 11 mutations) fall into three classes based on the architectural layer they modify:

### Proposer-Replacement Mutations
Replace or augment GEPA's reflective proposer while keeping the iteration loop, Pareto frontier, and minibatch sampling intact. Same rollout budget as GEPA (~3,600–7,000 rollouts).

| Method | Core Idea |
|--------|-----------|
| `gepa` | Paper-faithful baseline (ICLR 2026 Oral) |
| `best_of_k_K3` | Generate K=3 candidate prompts per round, keep the best |
| `contrastive_reflection` | Inject historical contrastive success/failure pairs into the reflection prompt |
| `failure_stratified_k_K3` | Best-of-K with partitioned failure coverage per candidate |
| `contrastive_synthesis` | Distill contrastive pairs into an abstract improvement principle via an extra LLM call |
| `active_minibatch` | Select high-variance (high-disagreement) examples for each reflection minibatch |

### Standalone Search Algorithms
Replace the GEPA loop entirely with a different search strategy. Typically use **far fewer rollouts** (25–1,500 vs 3,600–7,000).

| Method | Core Idea |
|--------|-----------|
| `synaptic_pruning` | One-shot: generate overspecified prompt, ablate sections, keep minimal effective subset |
| `tournament` | 64-candidate single-elimination bracket |
| `slime_mold` | Progressive colony pruning: 20→10→5→3→1 with inter-round mutation |
| `ant_colony` | Pheromone-based reinforcement over atomic prompt components |

### Structural / Curriculum Mutations
Keep GEPA as the optimizer but change how the training set is presented or how the prompt is structured.

| Method | Core Idea |
|--------|-----------|
| `ecological_succession` | Curriculum: easy→medium→hard examples across 3 GEPA phases |
| `modular` | Decompose prompt into 4 functional modules, optimize each via mini-GEPA, compose |

---

## Method Profiles

### Proposer-Replacement Mutations

These methods replace GEPA's proposal step while keeping the same outer loop and rollout budget.

---

#### GEPA (`gepa`) — Baseline

**Type:** Baseline (ICLR 2026 Oral)

**How it works:** 4-step iterative loop: trajectory sampling on minibatch of B=3 examples, natural language reflection by LLM, Pareto frontier selection to maintain diversity, and system-aware merge. This is the control against which all mutations are measured.

**Cost profile:** ~3,600–7,000 rollouts depending on benchmark. Wall clock 1.7–8.2h.

**Observed results (5-seed complete only):** 1.7B IFBench 0.062 (below 0.063 baseline), 8B IFBench 0.056 (below 0.068 baseline), 14B IFBench 0.097 (+76% over baseline — strongest here), 27B HotpotQA 0.848 (+2%), 27B PUPA 0.639 (+19%). Notably, **GEPA hurts performance on smaller models** (1.7B and 8B IFBench), suggesting the iterative reflection loop can degrade prompts when the underlying LM lacks the reasoning capacity to generate good reflections.

---

#### Best-of-K (`best_of_k`)

**Type:** Proposer-Replacement

**How it works:** Each iteration generates K=3 independent candidate prompts instead of one, evaluates all on the same minibatch, and selects the best. Brute-force diversification at the proposal step.

**Key difference from GEPA:** K× more LLM calls per iteration, no structural change to the loop.

**Cost profile:** ~3,600–7,000 rollouts (same as GEPA). ~K× more LLM calls. Wall clock 1.9–6.7h.

**Pros:**
- Strongest or near-strongest mutation across most benchmarks and scales
- Simple drop-in replacement; never fell below baseline in any completed run
- 27B HotpotQA: 0.923 (+11%), 8B IFBench: 0.101 (+48%), 1.7B PUPA: 0.524 (+59%)

**Cons:**
- K× LLM cost per iteration on expensive models
- No mechanism to steer diversity — candidates may collapse to similar proposals

**Observed results (5-seed complete only):** 6 completed cells. Always above baseline. Standouts: 27B HotpotQA 0.923, 8B IFBench 0.101, 1.7B PUPA 0.524, 14B PUPA 0.771.

---

#### Contrastive Reflection (`contrastive_reflection`)

**Type:** Proposer-Replacement

**How it works:** Searches historical candidates for ones that outperformed on specific examples, injects these contrastive pairs as side information into the reflection prompt. Zero extra LLM calls — mining is CPU-only.

**Key difference from GEPA:** Augments the reflector's context with mined success/failure contrasts.

**Cost profile:** ~3,600–4,000 rollouts (same as GEPA). Zero extra LLM calls. Wall clock 3.0–7.3h.

**Pros:**
- No additional LLM cost
- Solid on PUPA: 14B 0.766 (+13%), 27B 0.664 (+24%)

**Cons:**
- Requires enough history to mine meaningful contrasts — cold start issue
- Never the top method in any setting

**Observed results (5-seed complete only):** 4 completed cells. Always above baseline but never #1. Best: 14B PUPA 0.766, 27B PUPA 0.664.

---

#### Contrastive Synthesis (`contrastive_synthesis`)

**Type:** Proposer-Replacement

**How it works:** Extends Contrastive Reflection by adding a synthesis LLM call to distill mined contrastive pairs into an abstract improvement principle before injection.

**Key difference from GEPA:** Distilled principle rather than raw contrastive snippets.

**Cost profile:** Same rollouts as GEPA. 1 extra LLM call/iteration. Wall clock ~5.0h.

**Pros:**
- Strongest mutation on 1.7B HotpotQA: 0.729 (+13%)

**Cons:**
- Only 2 completed cells — limited evidence

**Observed results (5-seed complete only):** 1.7B HotpotQA 0.729, 1.7B IFBench 0.085. Both above baseline.

---

#### Active Minibatch (`active_minibatch`)

**Type:** Proposer-Replacement

**How it works:** Replaces uniform minibatch sampling with active learning — preferentially selects high-disagreement examples (70% active, 30% random).

**Key difference from GEPA:** Changes *which* examples the optimizer sees, not how it proposes.

**Cost profile:** Same rollouts as GEPA. Zero extra LLM calls. Wall clock ~3.6h.

**Pros:**
- 1.7B HotpotQA: 0.715 (+11%), 1.7B IFBench: 0.085 (+34%)
- No additional LLM cost

**Cons:**
- Only 2 completed cells

**Observed results (5-seed complete only):** 1.7B HotpotQA 0.715, 1.7B IFBench 0.085. Both above baseline.

---

### Standalone Search Mutations

These methods replace the entire GEPA loop with a different search strategy, typically using far fewer rollouts.

---

#### Synaptic Pruning (`synaptic_pruning`)

**Type:** Standalone Search

**How it works:** One-shot pipeline: generate 3 overspecified prompts, select best on 40-example validation, parse into sections, ablate each to identify negligible ones, prune, then strengthen load-bearing sections.

**Key difference from GEPA:** No iterative loop — a single generate-then-compress pipeline using 15–100× fewer rollouts.

**Cost profile:** **~25–460 rollouts** (vs 3,600–7,000 for GEPA). Wall clock **0.1–1.7h**.

**Pros:**
- Fastest method by far — 0.1h with ~25 rollouts on 1.7B IFBench
- 27B IFBench: 0.146 (+148% over baseline) — strongest result in the entire sweep
- 27B HotpotQA: 0.902 (+9%) in 1.7h vs GEPA's 0.848 in 8.2h

**Cons:**
- 8B IFBench: 0.064 — **below baseline**
- 27B PUPA: 0.521 — **below baseline**
- Fragile when the initial generation is poor — no recovery mechanism

**Observed results (5-seed complete only):** 9 completed cells. Above baseline in 7, below in 2. The most cost-efficient method when it works; harmful when it doesn't.

---

#### Tournament (`tournament`)

**Type:** Standalone Search

**How it works:** Generates 64 diverse candidates via 4 strategy-specific LLM calls, then single-elimination bracket over 6 rounds with increasing example counts.

**Key difference from GEPA:** Population-based selection without iterative refinement — relies on initial diversity.

**Cost profile:** ~1,000–1,500 rollouts (4–10× fewer than GEPA). Wall clock 1.3–3.1h.

**Pros:**
- 1.7B PUPA: 0.514 (+56%), 27B HotpotQA: 0.902 (+9%)
- Good efficiency — fewer rollouts with competitive results

**Cons:**
- 1.7B HotpotQA: 0.630 — **below baseline**
- 8B IFBench: 0.057 and 14B IFBench: 0.054 — **below baseline**
- No refinement after selection — ceiling limited by initial candidate quality

**Observed results (5-seed complete only):** 8 completed cells. Above baseline in 5, below in 3. Strong on PUPA and 27B; unreliable on IFBench at smaller scales.

---

#### Slime Mold (`slime_mold`)

**Type:** Standalone Search

**How it works:** Generates 20 candidates, progressively prunes over 4 rounds (20→10→5→3→1) with failure-informed mutation between rounds.

**Key difference from GEPA:** Progressive elimination with targeted mutation — combines population pruning with failure-driven refinement.

**Cost profile:** ~540–930 rollouts (8–12× fewer than GEPA). Wall clock 0.4–2.5h.

**Pros:**
- Dominates PUPA: 14B 0.785 (+16%), 27B 0.712 (+33%)
- 27B HotpotQA: 0.897 (+8%)

**Cons:**
- 14B IFBench: 0.011 — **catastrophic failure** (80% below baseline, worst result in the study)
- 27B IFBench: 0.057 — **below baseline**

**Observed results (5-seed complete only):** 9 completed cells. Above baseline in 6, below in 3. Best method for PUPA across scales. The 14B IFBench collapse to 0.011 is the single worst result in the entire sweep.

---

#### Ant Colony (`ant_colony`)

**Type:** Standalone Search

**How it works:** Decomposes prompts into ~50 atomic components across 5 categories. 50 rounds of pheromone-based search: 3 ants per round sample components, evaluate, reinforce high-scoring ones. Polish step composes top-15 into coherent prompt.

**Key difference from GEPA:** Component-level search rather than whole-prompt optimization.

**Cost profile:** ~1,500–1,800 rollouts. Wall clock ~1.6h.

**Pros:**
- 1.7B IFBench: 0.075 (+18%)

**Cons:**
- 8B IFBench: 0.067 — **below baseline**
- Only 2 completed cells — very limited evidence

**Observed results (5-seed complete only):** 1.7B IFBench 0.075 (above baseline), 8B IFBench 0.067 (below baseline). Insufficient data to draw conclusions.

---

### Structural / Curriculum Mutations

These methods change the structure of GEPA's optimization process rather than replacing components.

---

#### Ecological Succession (`ecological_succession`)

**Type:** Structural/Curriculum

**How it works:** Wraps GEPA in a curriculum: estimates example difficulty (3×N_train rollouts), then runs GEPA in 3 phases — easy only (15% budget), easy+medium (30%), all (55%).

**Key difference from GEPA:** Adds difficulty-aware curriculum ordering.

**Cost profile:** Same budget as GEPA + difficulty overhead. ~4,200 rollouts. Wall clock ~3.9h.

**Pros:**
- 1.7B IFBench: 0.071 (+13%)

**Cons:**
- Only 1 completed cell
- Modest gain for added complexity

**Observed results (5-seed complete only):** 1.7B IFBench 0.071. Above baseline but weakest mutation on that benchmark.

---

#### Modular (`modular`)

**Type:** Structural

**How it works:** Decomposes prompt into 4 functional modules, runs mini-GEPA on each independently (budget/5), LLM composition, then final joint refinement (budget/5).

**Key difference from GEPA:** Divide-and-conquer — optimizes components independently before composing.

**Cost profile:** Same total budget as GEPA. ~4,500 rollouts. Wall clock ~4.4h.

**Pros:**
- 1.7B IFBench: 0.080 (+27%)
- 1.7B HotpotQA: 0.661 (+3%)

**Cons:**
- Only 2 completed cells
- Composition step may lose synergies between modules

**Observed results (5-seed complete only):** 1.7B IFBench 0.080, 1.7B HotpotQA 0.661. Both above baseline.

---

## Benchmarks

| Benchmark | Task Type | Notes |
|-----------|-----------|-------|
| HotpotQA | Multi-hop QA | F1 scoring; requires 2+ step reasoning chains |
| HoVer | Fact verification | Binary SUPPORTS / NOT_SUPPORTED over structured evidence |
| PUPA | Privacy redaction | Rewrite queries replacing PII with `[REDACTED]` tags |
| IFBench | Instruction following | All constraints must be satisfied; partial credit only |
| LiveBench-Math | Math reasoning | AMC/AIME-difficulty problems, exact-match scoring |

AIME-2025 is in the codebase but excluded from the active sweep (insufficient timing data).

---

## Cluster Setup

Inference runs on a local SLURM cluster. Each model size is served independently via vLLM:

| Model | Node | GPU | VRAM | Port |
|-------|------|-----|------|------|
| Qwen3-27B-AWQ | manifold | RTX 5090 | 32 GB | 8124 |
| Qwen3-8B | archimedes | RTX 5090 | 32 GB | 8125 |
| Qwen3-4B | sapphire | RTX 4090 | 24 GB | 8126 |
| Qwen3-1.7B | sar-gpu-vm | RTX 3090 | 24 GB | 8127 |
| Qwen3-14B | kolmogorov | RTX 4090 | 24 GB | 8128 |

See [CLAUDE/cluster_infrastructure.md](CLAUDE/cluster_infrastructure.md) for full details, SLURM job chain setup, and downed node status.

---

## Quickstart

### Install

```bash
git clone --recurse-submodules <repo-url>
cd gepa-mutations-raycluster
uv sync
cp .env.example .env   # add TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID
```

### Launch a vLLM server

```bash
# Single job (ray-cluster partition — no time limit)
sbatch scripts/serve_vllm_8b.sh

# Job chain for time-limited partitions (capstone = 8h max)
J1=$(sbatch scripts/serve_vllm_27b_manifold.sh | awk '{print $4}')
J2=$(sbatch --dependency=afterany:$J1 scripts/serve_vllm_27b_manifold.sh | awk '{print $4}')
J3=$(sbatch --dependency=afterany:$J2 scripts/serve_vllm_27b_manifold.sh | awk '{print $4}')
```

### Smoke test (run first before every full sweep)

```bash
export GEPA_MODEL="Qwen/Qwen3-8B"
export GEPA_BASE_URL="http://10.0.10.58:8125/v1"
python scripts/run_all_local.py --smoke-test --workers 4
```

### Full sweep

```bash
export GEPA_MODEL="Qwen/Qwen3-8B"
export GEPA_BASE_URL="http://10.0.10.58:8125/v1"
python scripts/run_all_local.py --workers 8 --benchmark hotpotqa hover pupa ifbench livebench
```

The orchestrator is idempotent — it skips already-completed experiments on restart.

### Monitoring

```bash
# Per-model Telegram status (one message per model)
python scripts/monitor_multi_model.py --mode 15min

# Consolidated sweep health (one message total)
python scripts/monitor_multi_model.py --mode 30min
```

---

## Results (Multi-Model Sweep — In Progress)

**Status: 286/720 tasks complete (39.7%).** 4 model sizes × 3 benchmarks × 12 methods × 5 seeds = 720 total. Sweep started 2026-04-12; overnight cluster disruptions (3 mass-cancellations, GPU contention on inference node) reduced throughput. 14B-inference node unavailable; 14B runs limited to ansatz node only.

### Completion Matrix

| Model | HotpotQA | PUPA | IFBench | Total |
|-------|----------|------|---------|-------|
| **1.7B** | 29/60 | 21/60 | **55/60** | **105/180** (58%) |
| **8B** | 0/60 | 0/60 | 38/60 | 38/180 (21%) |
| **14B** | 0/60 | 28/60 | 24/60 | 52/180 (29%) |
| **27B-AWQ** | 30/60 | **30/60** | 32/60 | **92/180** (51%) |

### Key Findings

**Only results with all 5 seeds complete are used in comparisons below** (48 of 144 method/model/benchmark combinations). Partial results are excluded to ensure statistical validity.

**Prompt optimization provides meaningful lift across all model sizes.** Even on the smallest model (1.7B), the best optimization method improves over the raw baseline by +9pp on HotpotQA, +20pp on PUPA, and +3pp on IFBench.

| Model | Benchmark | Baseline | Best Method (5 seeds) | Score | Lift |
|-------|-----------|----------|-----------------------|-------|------|
| 1.7B | HotpotQA | 0.643 | Contrastive Synthesis | 0.729 | +0.085 |
| 1.7B | PUPA | 0.329 | Best-of-K | 0.524 | +0.195 |
| 1.7B | IFBench | 0.063 | Best-of-K | 0.096 | +0.034 |
| 8B | IFBench | 0.068 | Best-of-K | 0.101 | +0.032 |
| 14B | PUPA | 0.675 | Slime Mold | 0.785 | +0.110 |
| 14B | IFBench | 0.055 | GEPA | 0.097 | +0.042 |
| 27B-AWQ | HotpotQA | 0.830 | Best-of-K | 0.923 | +0.093 |
| 27B-AWQ | PUPA | 0.537 | Slime Mold | 0.712 | +0.175 |
| 27B-AWQ | IFBench | 0.059 | Synaptic Pruning | 0.146 | +0.087 |

### Detailed Results by Benchmark

Scores are mean `test_score` across 5 seeds (42, 123, 456, 789, 1024). Only combinations with **all 5 seeds complete** are shown. `—` = incomplete or not yet run. Baseline is a single evaluation of the seed prompt with no optimization.

#### HotpotQA (Multi-hop QA, F1 scoring)

| Method | 1.7B | 8B | 14B | 27B-AWQ |
|--------|------|------|------|------|
| Baseline (no opt) | 0.643 | 0.750 | 0.783 | 0.830 |
| GEPA | — | — | — | 0.848 |
| Best-of-K | — | — | — | **0.923** |
| Contrastive Synth. | **0.729** | — | — | — |
| Active Minibatch | 0.715 | — | — | — |
| Modular | 0.661 | — | — | — |
| Tournament | 0.630 | — | — | 0.902 |
| Synaptic Pruning | — | — | — | 0.902 |
| Slime Mold | — | — | — | 0.897 |

Best performers: **Best-of-K on 27B** (+9.3pp). **Contrastive Synthesis on 1.7B** (+8.5pp). Tournament underperforms baseline on 1.7B (-1.3pp).

#### PUPA (Privacy Redaction, PII-replacement accuracy)

| Method | 1.7B | 8B | 14B | 27B-AWQ |
|--------|------|------|------|------|
| Baseline (no opt) | 0.329 | 0.565 | 0.675 | 0.537 |
| GEPA | — | — | — | 0.639 |
| Best-of-K | **0.524** | — | 0.771 | 0.673 |
| Contrastive Refl. | — | — | 0.766 | 0.664 |
| Synaptic Pruning | 0.333 | — | 0.756 | 0.521 |
| Tournament | 0.514 | — | 0.714 | 0.590 |
| Slime Mold | 0.438 | — | **0.785** | **0.712** |

Best performers: **Slime Mold** dominates on 14B (+11.0pp) and 27B (+17.5pp). **Best-of-K** strongest on 1.7B (+19.5pp). **Synaptic Pruning hurts on 27B** (-1.6pp below baseline) — the pruning removes useful prompt content for this task.

#### IFBench (Instruction Following, constraint satisfaction)

| Method | 1.7B | 8B | 14B | 27B-AWQ |
|--------|------|------|------|------|
| Baseline (no opt) | 0.063 | 0.068 | 0.055 | 0.059 |
| GEPA | 0.062 | 0.056 | **0.097** | — |
| Best-of-K | **0.096** | **0.101** | — | — |
| Contrastive Refl. | 0.084 | 0.079 | — | — |
| Synaptic Pruning | 0.084 | 0.064 | 0.061 | **0.146** |
| Tournament | — | 0.057 | 0.054 | 0.083 |
| Slime Mold | 0.082 | 0.071 | 0.011 | 0.057 |
| Ant Colony | 0.075 | 0.067 | — | — |
| Active Minibatch | 0.085 | — | — | — |
| Contrastive Synth. | 0.085 | — | — | — |
| Ecological Succ. | 0.071 | — | — | — |
| Modular | 0.080 | — | — | — |

IFBench is the hardest benchmark — all scores below 15%. **Synaptic Pruning on 27B** is the standout (+8.7pp, the largest absolute lift in the sweep). **Best-of-K** consistent on smaller models. Notable failures: **GEPA hurts on 8B** (-1.2pp) and is flat on 1.7B — the iterative reflection loop can degrade prompts on weaker models. **Slime Mold catastrophically fails on 14B** (0.011 vs 0.055 baseline, -4.4pp).

### Methods That Hurt Performance

10 of 48 complete (5-seed) method/model/benchmark combinations score **below the unoptimized baseline**. Optimization is not always beneficial:

| Model | Benchmark | Method | Score | Baseline | Delta |
|-------|-----------|--------|-------|----------|-------|
| 1.7B | HotpotQA | Tournament | 0.630 | 0.643 | -0.013 |
| 1.7B | IFBench | GEPA | 0.062 | 0.063 | -0.001 |
| 8B | IFBench | GEPA | 0.056 | 0.068 | -0.012 |
| 8B | IFBench | Tournament | 0.057 | 0.068 | -0.011 |
| 8B | IFBench | Synaptic Pruning | 0.064 | 0.068 | -0.004 |
| 8B | IFBench | Ant Colony | 0.067 | 0.068 | -0.001 |
| 14B | IFBench | Slime Mold | 0.011 | 0.055 | **-0.044** |
| 14B | IFBench | Tournament | 0.054 | 0.055 | -0.001 |
| 27B-AWQ | IFBench | Slime Mold | 0.057 | 0.059 | -0.002 |
| 27B-AWQ | PUPA | Synaptic Pruning | 0.521 | 0.537 | -0.016 |

### Emerging Patterns

1. **No single method dominates.** Best-of-K wins on small models and HotpotQA/27B, Slime Mold on PUPA, Synaptic Pruning on IFBench/27B, Contrastive Synthesis on HotpotQA/1.7B.
2. **Model scale affects which methods work.** GEPA (the paper baseline) underperforms on 1.7B/8B IFBench but is the top method on 14B IFBench. Methods that avoid complex multi-step reasoning (Best-of-K, Tournament) tend to be more robust across scales.
3. **Optimization can hurt.** 10/48 complete combinations score below baseline — especially on IFBench where 8B models are particularly vulnerable. This underscores the importance of baseline comparisons.
4. **PUPA shows the largest lifts** (+10–20pp), likely because it's a rewriting task where prompt improvements translate directly to output quality.
5. **IFBench has a low ceiling** (<15% for all methods), suggesting instruction-following may require architectural rather than prompt-level improvements.

### Compute Cost vs GEPA Baseline

Standalone search methods (synaptic pruning, slime mold, tournament) use 5–150x fewer rollouts than GEPA while often achieving comparable or superior scores. This table compares cost for cases where both GEPA and the mutation have 5 complete seeds on the same model/benchmark:

| Model | Benchmark | Method | Lift vs Baseline | Rollouts | Wall Clock | GEPA Rollouts | GEPA Wall |
|-------|-----------|--------|-----------------|----------|------------|--------------|-----------|
| 1.7B | IFBench | Synaptic Pruning | +0.022 | **25** | 0.1h | 3,643 | 2.1h |
| 1.7B | IFBench | Best-of-K | **+0.034** | 3,655 | 3.8h | 3,643 | 2.1h |
| 1.7B | IFBench | Slime Mold | +0.019 | 930 | 2.2h | 3,643 | 2.1h |
| 8B | IFBench | Best-of-K | **+0.032** | 3,706 | 4.1h | 3,731 | 1.7h |
| 8B | IFBench | Synaptic Pruning | -0.004 | 460 | 0.8h | 3,731 | 1.7h |
| 14B | IFBench | GEPA | **+0.042** | 3,614 | 7.6h | — | — |
| 14B | IFBench | Synaptic Pruning | +0.006 | 460 | 0.9h | 3,614 | 7.6h |
| 27B | HotpotQA | GEPA | +0.018 | 6,873 | 8.2h | — | — |
| 27B | HotpotQA | Synaptic Pruning | +0.072 | **451** | 1.7h | 6,873 | 8.2h |
| 27B | HotpotQA | Best-of-K | **+0.093** | 7,029 | 3.9h | 6,873 | 8.2h |
| 27B | IFBench | Synaptic Pruning | **+0.087** | **460** | 0.6h | — | — |
| 27B | PUPA | Slime Mold | **+0.175** | **725** | 1.2h | 3,970 | 5.5h |
| 27B | PUPA | GEPA | +0.102 | 3,970 | 5.5h | — | — |

**Key takeaway:** Synaptic Pruning achieves the best score on 27B IFBench (+8.7pp) using only **460 rollouts in 0.6 hours** — compared to GEPA's typical 3,600–7,000 rollouts over 2–8 hours. This represents a **~15x reduction in compute** with *better* results. Standalone search methods (Synaptic Pruning, Slime Mold, Tournament) are consistently more cost-efficient than proposer-replacement methods that inherit GEPA's full rollout budget.

### Notes

- Baseline scores are from a single seed with the default seed prompt (no optimization). See `BENCHMARK_SEED_PROMPTS` in `src/gepa_mutations/runner/experiment.py`.
- 14B results are from a single node (ansatz) running with constrained GPU memory (`max-model-len=2048`). The inference node was unavailable due to GPU contention from another user. This may affect 14B scores — particularly the anomalous Slime Mold IFBench result (0.011).
- 8B has zero HotpotQA and PUPA results; 14B has zero HotpotQA results — these will be completed in the next sweep batch.
- Failed tasks from overnight cluster disruptions (no `result.json` written) will be automatically retried when orchestrators are relaunched.

---

## Configuration

Set in `.env`:

```
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
```

Set via environment before launching the orchestrator:

```bash
export GEPA_MODEL="Qwen/Qwen3-27B-AWQ"    # maps to runs/qwen3-27b-awq/
export GEPA_BASE_URL="http://10.0.10.69:8124/v1"
```

Paper hyperparameters and baseline scores are in `src/gepa_mutations/config.py`.

---

## Results Layout

```
runs/
  qwen3-27b-awq/
    hotpotqa/
      gepa/42/result.json
      best_of_k_K3/42/result.json
      ...
  qwen3-8b/
    ...
```

`result.json` key fields:
- `test_score` — final held-out accuracy
- `train_scores` — per-round training scores  
- `elapsed` — wall-clock seconds

---

## Repository Structure

```
gepa/                    GEPA submodule (v0.1.1, patched — see CLAUDE/known_bugs_and_fixes.md)
src/gepa_mutations/      Shared infrastructure
  benchmarks/            Dataset loaders and evaluators
  runner/                Experiment runner, LM wrapper, callbacks
  notifications/         Telegram alerts
  config.py              Settings and paper baseline scores
methods/                 Algorithm mutations (one editable package each)
scripts/
  run_all_local.py       Parallel experiment orchestrator
  monitor_multi_model.py Telegram monitoring (15-min and 30-min modes)
  check_node_recovery.sh Cron: alerts when downed nodes recover
  smoke_test_all.py      Pre-sweep smoke test runner
  serve_vllm_*.sh        SLURM job scripts for each vLLM worker
CLAUDE/                  Operational knowledge for AI assistants
docs/                    Planning docs, mutation selection report
configs/                 Experiment configurations
notebooks/               Analysis notebooks
tests/                   Test suite
data/                    Dataset cache (raw/ gitignored)
runs/                    Results (gitignored)
logs/                    Logs (gitignored)
```

---

## Known Issues

See [CLAUDE/known_bugs_and_fixes.md](CLAUDE/known_bugs_and_fixes.md):

- **gepa state save `FileNotFoundError` on NFS** — patched in `gepa/src/gepa/core/state.py`
- **vLLM IPC socket path length limit** — fix: `cd /tmp` before launching (all serve scripts do this)
- **`_env_model_tag()` 14B/4B substring collision** — fixed in `scripts/run_all_local.py`
- **`tournament` invisible to monitoring** — method writes no intermediate files; use vLLM `/metrics` to confirm activity
