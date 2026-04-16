# Experiment 01: SLURM Sweep (Mac + Cluster)

**Dates:** 2026-03 (approximate)
**Status:** complete (archived)
**Models:** qwen3-1.7b, qwen3-4b, qwen3-8b, qwen3-14b, qwen3-27b-awq
**Deployment:** SLURM cluster (multiple GPU types) + MacBook Pro M5 Max MLX

## Purpose
Initial large-scale sweep across 4-6 benchmarks × 6 methods × 5 seeds on 5 model sizes.
Established baselines and identified Slime Mold's high ceiling on HotpotQA with large variance.

## Methods Tested
baseline, GEPA, contrastive_reflection, synaptic_pruning, slime_mold, tournament (and others)

## Benchmarks × Seeds
6 benchmarks × 5 seeds. See runs/ for complete data.

## Key Findings
- Slime Mold has highest ceiling on HotpotQA (0.719 on qwen3-1.7b) but with huge variance
- Most methods plateau by ~1,500 LLM calls
- Paper GEPA scores (Qwen3-8B): HotpotQA 62.33%, IFBench 38.61%, HoVer 52.33%, PUPA 91.85%

## Cost
Internal SLURM + local Mac.
