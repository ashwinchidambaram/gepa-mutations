# Idea: Larger Initial Pool for Slime Mold

## Current State
- 4 strategies × 5 prompts = 19 generated + 1 seed = **20 candidates**
- Cascade: 20 → 10 → 5 → 3 → 1
- ~540 rollouts

## Proposal
Increase to **40 candidates** (still well under GEPA's 2,000-7,000 budget):
- 4 strategies × 10 prompts = 39 generated + 1 seed = 40
- Or 6 strategies × 7 prompts (add "Constraint-focused" and "Example-heavy")

## New Cascade Options

**Option A: 5 rounds (one extra round)**
```
40 → 20 → 10 → 5 → 3 → 1
R1: 40×10 = 400
R2: 20×15 = 300
R3: 10×20 = 200
R4:  5×25 = 125
R5:  3×30 =  90
Total: ~1,115 rollouts
```

**Option B: 4 rounds (bigger cuts)**
```
40 → 15 → 5 → 3 → 1
R1: 40×10 = 400
R2: 15×15 = 225
R3:  5×20 = 100
R4:  3×30 =  90
Total: ~815 rollouts
```

## Why
- More initial diversity = better chance the "right direction" is in the pool
- More signal in early rounds (40 on 10 examples > 20 on 10 examples)
- Mutation compounds better with more survivors
- Still 2-6x cheaper than GEPA even at 40 candidates

## Cost Comparison
| Pool Size | Rollouts | vs GEPA (avg ~4,500) |
|-----------|----------|---------------------|
| 20 (current) | ~540 | 8x cheaper |
| 40 (option A) | ~1,115 | 4x cheaper |
| 40 (option B) | ~815 | 5.5x cheaper |
| 64 (tournament-sized) | ~1,680 | 2.7x cheaper |
