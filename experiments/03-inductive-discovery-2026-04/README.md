# Experiment 03: Inductive Strategy Discovery for Slime Mold

**Dates:** 2026-04-16 onward
**Status:** active
**Model:** qwen3-8b (bf16)
**Deployment:** RunPod, 1x A40 48GB (TBD)

## Purpose
Test whether inductive strategy discovery (LLM examines benchmark examples and identifies task-specific skills, then generates specialized prompts per skill) produces better candidates than personality-based strategies. Headline hypothesis: reduces seed-level variance while maintaining Slime Mold's high ceiling.

## Spec
See `../../docs/superpowers/specs/2026-04-15-inductive-strategy-discovery-design.md`

## Methods Tested (Tier 1)
1. slime_mold (baseline — 4 personality strategies, blind mutation)
2. slime_mold_prescribed8 (8 universal problem-solving strategies, blind mutation)
3. slime_mold_inductive_k5 (inductive discovery K=5, blind mutation)
4. slime_mold_inductive_k5_crosspollin (inductive + cross-pollination mutation)
5. slime_mold_inductive_k5_refresh_expand (full hybrid + refresh pass)

## Benchmarks × Seeds
hotpotqa × 10 seeds + {hover, pupa, ifbench} × 5 seeds = 125 runs
Seeds: 42, 123, 456, 789, 1024 (+ 2048, 4096, 8192, 16384, 32768 for HotpotQA variance)

## Key Findings
(to be filled in when experiment completes)

## Cost
~$30 on 1x A40 (~63 hours est.)
