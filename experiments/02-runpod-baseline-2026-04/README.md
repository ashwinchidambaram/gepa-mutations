# Experiment 02: RunPod Baseline Sweep

**Dates:** 2026-04 (approximate)
**Status:** complete (archived)
**Models:** qwen3-1.7b (Pod A), qwen3-8b (Pod B)
**Deployment:** RunPod, 2 pods (RTX 3090 Ti + RTX 5000 Ada)

## Purpose
Re-run Slime Mold and related methods on RunPod infrastructure with fixed evaluators (IFBench and HoVer critical bug fixes).
Validated that variance and ceiling findings hold post-fix.

## Methods Tested
baseline, slime_mold, tournament, synaptic_pruning, contrastive_reflection, gepa

## Benchmarks × Seeds
hotpotqa, hover, pupa, ifbench, livebench × 5 seeds (42, 123, 456, 789, 1024)

## Cost
~$30 RunPod GPU time (2 pods, ~2 days each)

## Key Findings
Fed into Experiment 03 design:
- Slime Mold variance on HotpotQA confirmed as #1 opportunity
- GEPA call counts measured: ~4,000 per run on 8B
- Slime Mold call counts measured: ~650-1,250 per run on 8B (3-6x cheaper than GEPA)
