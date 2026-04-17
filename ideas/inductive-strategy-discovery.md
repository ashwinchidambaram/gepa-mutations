# Proposal: Inductive Strategy Discovery (Hybrid Model)

## Problem

Both the current personality-based strategies (Analytical, Creative, Minimal, Expert) and the proposed failure-mode-oriented strategies (Task Decomposition, Constraint-First, etc.) are **prescribed top-down**. We decide what strategies exist before seeing the task. This means:

1. The strategies may not match the actual failure modes of a given benchmark
2. Two benchmarks with very different skill requirements get the same strategy set
3. We're guessing at what matters instead of letting the data tell us

## Proposal: Two-Phase Hybrid Generation

### Phase 1: Strategy Discovery (1 LLM call)

Give the LLM a sample of benchmark examples and ask it to **inductively identify** the distinct skills/capabilities required:

```
Here is the task description and N examples from the benchmark.

1. Examine these examples carefully
2. Identify the K distinct skills, reasoning patterns, or capabilities 
   required to solve them well
3. For each skill, note which examples stress-test it most
4. For each skill, describe what FAILURE on that skill looks like
5. Propose K generation strategies, each targeting a different skill
```

The LLM outputs something like:
```
Skill 1: "Multi-hop entity tracking" — questions 2, 4 require chaining 
         facts across 3 paragraphs. Failure: stops at first hop.
Skill 2: "Distractor resistance" — questions 3, 5 include irrelevant 
         paragraphs. Failure: pulls answer from wrong paragraph.
Skill 3: "Implicit relationship inference" — question 1 requires inferring 
         population from a separate article. Failure: answers only from 
         the paragraph that mentions the entity.
...
```

### Phase 2: Strategy-Specialized Generation (K LLM calls)

For each discovered skill, generate prompts that **specialize** in that skill while covering the basics:

```
You are writing a system prompt for an AI assistant on this benchmark.

The core skill this prompt must EXCEL at: {discovered_skill}
What failure looks like: {failure_description}
Examples that test this skill hardest: {stress_test_examples}

Write a system prompt that:
1. Handles the general task competently
2. Has specific, structural techniques for {discovered_skill}
3. Would score highest on examples requiring {discovered_skill},
   even if it's slightly weaker on other dimensions
```

### Why Hybrid > Pure Inductive

**Pure inductive** (run discovery N times, hope for diversity) risks convergence — the LLM identifies the same skills every time.

**Pure prescribed** (our 6 fixed strategies) risks irrelevance — the strategies may not match the task.

**Hybrid** gets the best of both:
- **Task-adaptive**: strategies emerge from the actual benchmark data
- **Guaranteed diversity**: each prompt specializes in a DIFFERENT discovered skill
- **Structured**: Phase 1 produces an explicit skill decomposition that can be logged, compared, and reused

### Optional: Multi-Perspective Discovery

Run Phase 1 multiple times (2-3 calls) with different framing to get different skill decompositions:

```
# Perspective A: "What skills does this require?"
# Perspective B: "What are the common failure modes?"
# Perspective C: "If you had to train 6 specialist models, what would each specialize in?"
```

Then deduplicate across perspectives. Skills identified by multiple perspectives are high-confidence; unique ones add diversity.

## Interaction with Strategy-Aware Mutation

This composes naturally with strategy-aware mutation (see `strategy-aware-mutations.md`):

1. **Discovery** identifies skills and failure modes from examples
2. **Generation** creates prompts specialized per skill
3. **Mutation** knows the skill specialization → can make targeted improvements
4. **Cross-pollination** can incorporate techniques from other skill specialists

The mutation prompt becomes even more targeted:
```
This prompt specializes in MULTI-HOP ENTITY TRACKING.
It failed on examples requiring DISTRACTOR RESISTANCE.

Try incorporating distractor-handling techniques while 
preserving the multi-hop tracking structure.
```

## Implementation

### Changes to generation pipeline:

```python
def discover_strategies(reflection_lm, task_description: str, 
                        examples: list[dict], k: int = 6) -> list[Strategy]:
    """Phase 1: LLM examines examples, identifies K distinct skills."""
    discovery_prompt = f"""
    Task: {task_description}
    
    Examples:
    {format_examples(examples)}
    
    Identify {k} distinct skills required. For each:
    - Name (2-3 words)
    - Description (what capability this is)
    - Stress-test examples (which examples require this most)
    - Failure pattern (what going wrong looks like)
    """
    response = reflection_lm(discovery_prompt)
    return parse_strategies(response)  # structured output


def generate_specialized_prompt(generation_lm, strategy: Strategy,
                                task_description: str, 
                                examples: list[dict]) -> Candidate:
    """Phase 2: Generate a prompt that specializes in one skill."""
    gen_prompt = f"""
    Write a system prompt for: {task_description}
    
    This prompt must EXCEL at: {strategy.name}
    What failure looks like: {strategy.failure_pattern}
    Hardest examples for this skill: {strategy.stress_examples}
    
    The prompt should handle the general task well, but have specific
    structural techniques for {strategy.name}.
    """
    text = generation_lm(gen_prompt)
    return Candidate(
        text=text,
        strategy=strategy.name,
        strategy_source="inductive",  # vs "prescribed"
        skill_description=strategy.description,
    )
```

### Changes to `colony.py` / `runner.py`:

```python
# Before generating candidates:
strategies = discover_strategies(
    reflection_lm=lm,
    task_description=benchmark.description,
    examples=random.sample(benchmark.train, 10),  # sample for discovery
    k=6,
)

# Generate candidates, one per strategy
candidates = []
for strategy in strategies:
    for _ in range(prompts_per_strategy):
        candidate = generate_specialized_prompt(
            generation_lm=lm,
            strategy=strategy,
            task_description=benchmark.description,
            examples=random.sample(benchmark.train, 5),
        )
        candidates.append(candidate)
```

## Cost

**Phase 1 (discovery):** 1-3 extra LLM calls (negligible vs generation/evaluation budget)
**Phase 2 (generation):** Same number of calls as current approach
**Total overhead:** ~1-3 LLM calls. Essentially free.

## Evaluation Plan

### Hypothesis
Inductive strategy discovery will:
1. Produce strategies more relevant to each benchmark than prescribed strategies
2. Produce candidates with MORE DIVERSE failure patterns (measurable via failure overlap)
3. Produce higher champion scores (the "right" strategies are in the pool)

### Metrics
- **Strategy relevance**: Do discovered strategies name capabilities that actually differentiate high/low scorers?
- **Failure overlap**: Compare pairwise failure overlap between inductive vs prescribed strategies
- **Discovery stability**: How much do discovered strategies vary across seeds/runs?
- **Champion score**: Test set performance vs prescribed strategies vs personality-based baseline

### Convergence Tracking

Every method must log `(cumulative_llm_calls, best_score_so_far)` at every step — not just final scores. This produces convergence curves that answer:

- At what budget does each method plateau?
- Does strategy discovery raise the ceiling, the convergence speed, or both?
- At equal budgets, which method has the higher score?

**Why this matters:** If our budget is too small, we might stop before any method converges and draw wrong conclusions. Convergence curves let us verify the experiment was valid.

**Methodology:**
1. **Pilot run** (1 seed, generous budget ~2x planned) to identify roughly where each method flattens
2. **Set experiment budget** comfortably past the latest convergence point from the pilot
3. **Full experiment** — log every step, plot curves, verify flattening
4. **Validity check:** If a method's improvement rate hasn't dropped below 1% of current score by end of budget, flag that run as "may not have converged" — the result is inconclusive for that seed
5. **Report honestly:** "4/5 Slime Mold seeds converged by call 500; 1 was still improving" is useful data

**What to log per step:**
```python
{
    "method": "iso_inductive",
    "seed": 42,
    "step": 3,                    # round number or iteration
    "cumulative_llm_calls": 340,  # ALL calls: generation + mutation + evaluation
    "best_score": 0.78,
    "best_candidate_id": "c_14",
    "improvement_from_last": 0.03,
    "num_survivors": 5,
}
```

**Convergence criterion (for validity, not early stopping):**
A run is considered converged if `improvement_from_last < 0.01` for 2 consecutive steps. Runs that don't meet this by end of budget are flagged.

### Experiment
Run on 3 benchmarks × 5 seeds:
1. Personality-based strategies (current baseline)
2. Prescribed failure-mode strategies (from strategy-aware-mutations.md)
3. Inductive discovery (this proposal)
4. Inductive discovery + strategy-aware mutation (full hybrid)

## Applies To

- **Slime Mold**: Full hybrid (discovery + specialized generation + strategy-aware mutation)
- **Tournament**: Discovery + specialized generation (no mutation)
- **Best-of-K**: Discovery could generate the K candidates per iteration
- **GEPA baseline**: Discovery could inform the reflection prompt's understanding of task skills

## Adaptive K

Instead of fixing K=6 (or 8), let the LLM decide how many distinct skills it finds:

```
Identify the distinct skills required to solve these examples well.
There may be as few as 3 or as many as 10 — report however many you 
actually find. Don't pad the list to hit a number, and don't artificially 
merge skills that are genuinely distinct.
```

**Risk:** LLMs tend to converge on 5-6 because that "feels like a good answer." Validate that the count actually varies across benchmarks. If it doesn't, fixed K with a diverse set may be more reliable.

**Mitigation:** Run discovery 3 times with different framings (see Multi-Perspective Discovery above). Take the union of all identified skills, deduplicate, and let the resulting count be your K.

## Refresh Discovery Between Rounds

After Round 1 evaluation, you have real data: examples that ALL strategies struggled with. Re-run discovery focused on these failures:

```
The following examples were difficult for every strategy we tried.
None of our current skill-specialized prompts scored well on them.

{hard_examples}

What skills do these examples require that weren't captured by 
these existing strategies?
{list_of_current_strategies}

Identify 1-3 NEW skills that would specifically address these gaps.
```

This surfaces skills the first pass missed — either because the initial sample didn't include enough hard examples, or because the skills are subtle and only emerge when you see what fails.

**When to refresh:**
- After Round 1: cheapest, earliest signal
- Between any round where >30% of failures are shared across all survivors
- Don't refresh every round — diminishing returns after 1-2 refreshes

## Prescribed Strategy Baseline (8 Universal Problem-Solving Strategies)

To compare against inductive discovery, we also test with 8 prescribed universal strategies (see `strategy-aware-mutations.md` for full details):

1. **Divide & Conquer** — break into sub-problems, solve easiest first, assemble
2. **Working Backward** — start from end state, reason backward to start
3. **Analogy / Pattern Matching** — map onto known solved problems
4. **Abstraction** — strip specifics, solve simplified version, re-add detail
5. **Root Cause Analysis** — ask "why?" iteratively before solving
6. **Trial & Error** — systematically hypothesize, test, refine
7. **Constraint Satisfaction** — enumerate all rules, solve within narrowed space
8. **Lateral Thinking** — challenge assumptions, reframe, explore unconventional approaches

These are domain-agnostic problem-solving strategies rather than LLM-specific prompting techniques. The hypothesis is that they produce structurally different reasoning approaches regardless of the benchmark.

## Open Questions

1. **How many examples for discovery?** Too few → shallow skills. Too many → expensive context. 5-10 seems right.
2. **Should discovery use train or val examples?** Train avoids data leakage into strategy design.
3. **Does adaptive K actually vary?** Need to validate empirically that different benchmarks produce different K values.
4. **How many refreshes?** Probably 1-2 max. Diminishing returns after that.
5. **Inductive vs prescribed vs hybrid?** The experiment will tell us, but the hybrid (inductive discovery + prescribed fallback for any missing universal strategies) may be the most robust.
