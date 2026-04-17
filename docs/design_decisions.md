# Design Decisions — RosettaStone (Framing B)

Rationale for non-obvious methodology choices. Logged at decision time
to preserve reasoning while fresh. Required for paper-writing.

---

## 1. Rollouts capped, reflection calls tracked+weighted (not capped)

**Decision:** The matched-budget condition caps *task-model evaluation calls*
(rollouts) at a fixed value (e.g., 1000), but allows reflection/proposal calls
to be unlimited — they are tracked and factored into cost via the parameter-weighted
cost model.

**Rationale:** Capping reflections would require defining an equivalent
"reflection budget" across methods with fundamentally different reflection
patterns. GEPA makes many small reflection calls per iteration. MIPROv2
makes few large proposal calls upfront. ISO makes discovery + mutation calls.
A fixed reflection cap would distort each method differently. Instead, we
measure total cost (rollouts + weighted reflections) and plot on the Pareto
x-axis, letting readers see how much "thinking" each method needed.

**Risk:** A reviewer could argue this is itself a confound — GEPA might
benefit from unlimited reflection while ISO doesn't. We address this by
reporting both parameter-weighted and token-weighted costs as sensitivity
analysis.

---

## 2. Kendall's τ replaced ANOVA significance in falsification criterion #1

**Decision:** Falsification criterion #1 uses Kendall's τ ≥ 0.8 between
matched-budget and natural-budget method rankings (on ≥2/3 benchmarks),
not an ANOVA significance test on the method×analyzer interaction.

**Rationale:** With n=3 seeds per cell, ANOVA has extremely low power for
detecting medium-effect interactions. A significance threshold would bias
toward false falsification — the protocol would almost always be "falsified"
not because rankings truly match, but because p-values are inflated by tiny
sample sizes. Kendall's τ is a direct measure of rank agreement and is
interpretable at any sample size.

**Threshold justification:** τ ≥ 0.8 means near-perfect rank agreement
between matched and natural budgets. For 3 methods, the possible τ values
are {-1, -1/3, 1/3, 1}, so τ ≥ 0.8 effectively requires τ = 1.0 (identical
rankings). This is a conservative test — we need perfect rank agreement to
claim the confound doesn't exist.

---

## 3. MIPROv2 natural budget — determination pending

**Decision:** PENDING. MIPROv2's evaluation budget is not a single fixed
number — it depends on:
- `auto` mode: "light" (~720 evals), "medium" (~2100), "heavy" (~6600)
- Number of predictors in the program
- Whether minibatch evaluation is enabled
- Validation set size

**Default configuration (DSPy MIPROv2):**
- `auto="light"`, `val_size=100`, `minibatch=True`, `minibatch_size=35`
- For 1 predictor with demos: ~720 total metric evaluations

**Action needed:** Determine which mode the GEPA paper used for their
MIPROv2 comparison. If unspecified, default to "light" mode and document.
The GEPA paper's `PAPER_ROLLOUTS` only has entries for GEPA and GRPO,
not MIPROv2 — suggesting MIPROv2 was run at DSPy defaults.

**Working assumption:** MIPROv2 natural budget = DSPy "light" defaults
(~720 rollouts for single-predictor tasks, ~1200 without minibatch).
This is provisional until verified against the GEPA paper's methods section.

---

## 4. ISO natural budget — derived from method design parameters

**Decision:** PENDING. ISO has no publication defining a "natural" budget.
Must derive from method design parameters:

**Design-default budget computation:**
- Initial pool: 21 candidates (5 strategies × 4 prompts + seed)
- Round 1: 21 candidates × 10 examples = 210 rollouts
- Round 2: 10 survivors × 15 examples = 150 rollouts
- Round 3: 5 survivors × 20 examples = 100 rollouts
- Round 4: 3 survivors × 30 examples = 90 rollouts
- Mutation: ~5 calls per round × 3 rounds = ~15 reflection calls
- Discovery: 1-2 reflection calls
- Full val evaluation of champion: ~300 rollouts
- Full test evaluation: ~300 rollouts
- **Total: ~850 rollouts + ~17 reflection calls**

For `iso_refresh` (with refresh pass after R1): add ~8 new candidates
+ partial re-evaluation → ~1050-1150 total rollouts.

**Risk:** Since we define ISO's budget ourselves, any choice can be
criticized as cherry-picked. Mitigation: commit the derivation to git
before running natural-budget experiments, and note it transparently
in the paper's methods section.
