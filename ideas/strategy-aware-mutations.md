# Proposal: Strategy-Aware Generation & Mutation

## Problem

Current generation strategies (Analytical, Creative, Minimal, Expert for Slime Mold; Reasoning, Format, Detail, Error Prevention for Tournament) describe **personality traits or surface characteristics**, not functionally different approaches to the task. Two "creative" prompts and two "analytical" prompts might fail on the exact same examples — they sound different but don't produce meaningfully different behavior.

Additionally, mutations are blind. When a survivor gets mutated, the mutation prompt is generic: "improve this prompt." It has no idea what strategy generated the prompt, what the prompt's structural theory is, or what axis to improve along.

## Proposal: Two Changes

### 1. Failure-Mode-Oriented Generation Strategies

Replace personality-based strategies with strategies that target **different failure modes**:

| Strategy | What it generates | Failure mode it targets |
|----------|-------------------|------------------------|
| **Divide & Conquer** | Prompts that break problems into smaller sub-problems, solve easiest first, assemble | Fails when sub-problems are tightly coupled or decomposition loses holistic context |
| **Working Backward** | Prompts that start from the desired end state and reason backward to the current state | Fails when the end state is ambiguous or there are multiple valid endpoints |
| **Analogy / Pattern Matching** | Prompts that map the current problem onto known solved problems from other domains | Fails when the problem is genuinely novel with no structural analogues |
| **Abstraction** | Prompts that strip away specifics to find core logic, solve the simplified version, then re-add detail | Fails when the "messy details" ARE the problem (edge cases, format constraints) |
| **Root Cause Analysis** | Prompts that ask "why?" iteratively (5 Whys) to find the fundamental issue before solving | Fails on tasks that are generative rather than diagnostic (no "root cause" to find) |
| **Trial & Error** | Prompts that systematically generate hypotheses, test against constraints, and refine | Fails on tasks with no feedback signal or where first-shot accuracy matters |
| **Constraint Satisfaction** | Prompts that enumerate all rules/limits/constraints upfront, then solve within the narrowed space | Fails when constraints interact in non-obvious ways or when creative solutions require relaxing constraints |
| **Lateral Thinking** | Prompts that challenge assumptions, reframe the problem, and explore unconventional approaches | Fails on straightforward tasks where the obvious approach is correct |

**Why this is better:** Each strategy produces prompts with a **different structural theory** about how to solve the task. When they fail, they fail on DIFFERENT examples — which is exactly what you want for:
- Tournament: competition finds the best among genuinely diverse approaches
- Slime Mold: mutation can learn from different failure patterns

### 2. Strategy-Aware Mutation (Slime Mold Only)

Track which strategy generated each candidate. When mutating survivors, provide this context:

**Current (blind) mutation prompt:**
```
You are improving a system prompt for an AI assistant.

Current prompt:
{prompt}

This prompt achieved a score of 0.65. It failed on these examples:
{failures}

Improve this prompt to fix the failures while keeping what already works.
```

**Proposed (strategy-aware) mutation prompt:**
```
You are improving a system prompt for an AI assistant.

Current prompt:
{prompt}

This prompt was generated using a TASK DECOMPOSITION strategy — it breaks
problems into sub-steps. It achieved a score of 0.65.

It failed on these examples:
{failures}

The failures suggest {failure_pattern_hint}.

Options:
1. Improve the task decomposition approach to handle these cases
2. Incorporate elements from other strategies (e.g., add error-prevention
   guards, or add self-verification for ambiguous cases)

Improve the prompt. Keep what works, fix what doesn't.
```

### 3. Cross-Pollination During Mutation (Advanced)

When a prompt from strategy A fails, explicitly suggest incorporating ideas from strategy B:

```
This TASK DECOMPOSITION prompt fails on examples where constraints interact.
A CONSTRAINT-FIRST approach might handle these better.

Try incorporating constraint-checking into the decomposition structure:
- Keep the step-by-step breakdown
- Add a constraint-verification step after each sub-step
```

This is analogous to GEPA's component-level surgical edits, but at the strategy level instead of the text level.

## Implementation

### Changes to `colony.py`:

```python
# Track strategy origin per candidate
@dataclass
class Candidate:
    text: str
    strategy: str  # "task_decomposition", "constraint_first", etc.
    generation: int  # which round created this version
    parent_strategy: str | None  # if mutated, what was the parent's strategy

# 8 universal problem-solving strategies
GENERATION_STRATEGIES = [
    ("divide_and_conquer", "Generate a prompt that breaks the problem into smaller sub-problems, solves easiest first, and assembles results"),
    ("working_backward", "Generate a prompt that starts from the desired end state and reasons backward step by step to the current state"),
    ("analogy", "Generate a prompt that maps the problem onto known solved patterns and adapts existing solutions"),
    ("abstraction", "Generate a prompt that strips away specifics to find core logic, solves the simplified version, then re-adds detail"),
    ("root_cause_analysis", "Generate a prompt that asks 'why?' iteratively to find the fundamental issue before attempting a solution"),
    ("trial_and_error", "Generate a prompt that systematically generates hypotheses, tests them against constraints, and refines"),
    ("constraint_satisfaction", "Generate a prompt that enumerates all rules and limits upfront, then solves within the narrowed space"),
    ("lateral_thinking", "Generate a prompt that challenges assumptions, reframes the problem, and explores unconventional approaches"),
]

# Strategy-aware mutation
def mutate_prompt(reflection_lm, candidate: Candidate, failures: list[dict]) -> Candidate:
    mutation_prompt = f"""
    You are improving a system prompt.

    Current prompt (strategy: {candidate.strategy}):
    {candidate.text}

    Score: {candidate.score:.3f}. Failed on:
    {format_failures(failures)}

    This prompt uses a {candidate.strategy} approach.
    Improve it — either refine the current strategy or incorporate
    elements from other approaches to fix the failures.
    """
    improved_text = reflection_lm(mutation_prompt)
    return Candidate(
        text=improved_text,
        strategy=candidate.strategy,  # preserve lineage
        generation=candidate.generation + 1,
        parent_strategy=candidate.strategy,
    )
```

### Changes to `runner.py`:

- Store strategy metadata alongside each candidate
- Pass strategy to `mutate_prompt()` calls
- Log strategy lineage in metrics for analysis:
  - Which strategies produce the most survivors?
  - Do cross-pollinated mutations outperform same-strategy mutations?
  - Does the champion's lineage trace back to one dominant strategy?

## Rollout Cost

No change. This only affects:
- The generation prompt text (same number of LLM calls)
- The mutation prompt text (same number of LLM calls)
- Some metadata tracking (CPU only)

The diversity improvement is FREE in terms of rollouts.

## Evaluation Plan

### Hypothesis
Strategy-aware generation + mutation will:
1. Produce candidates that fail on MORE DIVERSE examples (measurable via failure overlap between strategies)
2. Produce higher champion scores (measurable on test set)
3. Show clearer strategy lineage patterns (which strategies tend to win?)

### Metrics to Track
- **Failure overlap**: For each pair of strategies, what % of failures are shared? Lower = more diverse = better.
- **Strategy survival rate**: Which strategies produce the most Round 4 survivors?
- **Cross-pollination success**: When mutation incorporates another strategy, does the score improve more than same-strategy mutation?
- **Champion lineage**: Which strategy does the champion trace back to? Is it consistent across seeds?

### Convergence Tracking

Every method must log `(cumulative_llm_calls, best_score_so_far)` at every step to produce convergence curves. This is critical for honest comparison with GEPA.

**What to log per step:**
```python
{
    "method": "iso_strategy_aware",
    "seed": 42,
    "step": 3,                    # round number
    "cumulative_llm_calls": 340,  # ALL calls: generation + mutation + evaluation
    "best_score": 0.78,
    "improvement_from_last": 0.03,
}
```

**Methodology:**
1. **Pilot run** (1 seed, 2x planned budget) to find where each method flattens
2. **Set experiment budget** past the latest convergence point
3. **Full experiment** — log every step, plot convergence curves
4. **Validity check:** If improvement hasn't dropped below 1% of current score for 2 consecutive steps by end of budget, flag the run as "may not have converged"

This lets us compare: at equal budgets, which method scores higher? And does each method actually converge within the budget, or are we cutting it off too early?

### Experiment
Run on 2 benchmarks (HotpotQA + IFBench) × 5 seeds:
- Current slime mold (baseline)
- New strategies, blind mutation (isolate strategy effect)
- New strategies, strategy-aware mutation (full proposal)

Compare champion test scores, failure diversity, and convergence curves.

## Applies To

- **Slime Mold**: full proposal (strategy-aware generation + mutation)
- **Tournament**: strategy-aware generation only (no mutation to improve)
- **Best-of-K**: could use strategy-aware generation for the K candidates per iteration
- **GEPA baseline**: doesn't generate a pool, so generation strategies don't apply — but the failure-mode thinking could inform the reflection prompt
