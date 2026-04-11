# Novel Prompt Optimization Ideas

30 experimental methodologies for prompt optimization, designed to beat GEPA on cost, speed, and quality. Organized in three sets based on how they were derived.

**Evaluation criteria (all ideas measured against GEPA baseline):**
- Cost less compute (fewer total rollouts)
- Are overall cheaper (fewer LLM calls)
- Are faster at optimization (lower wall-clock time)
- Are better at prompt optimization across all benchmarks

**GEPA baseline per iteration:** 2N rollouts (N with traces + N without) + 1 reflection LLM call. For aime (7051 budget), that's ~3500 iterations.

**Constraint:** All methods must use a single model (no multi-model cascades or ensembles of different model sizes).

---

## Top 15 Most Promising Ideas

The following 15 ideas (marked with a star in their titles below) were selected as the most promising to experiment with:

**From Set 1 (experiment-informed):**
- #1 Evaluation-Free Screening (EFS)
- #2 Active Minibatch Selection (AMS)
- #4 Contrastive Synthesis Reflection (CSR)
- #6 Failure-Clustered Targeted Mutation (FCTM)
- #7 Reflection Trajectory Caching (RTC)

**From Set A (novel independent):**
- A3 Tournament Selection (PTS)
- A4 Adversarial Minimax (AMPO)
- A5 Modular Decomposition (PDMO)
- A8 Constraint/Principle Optimization (CFPO)
- A10 Dual Critic-Guided (DCGO)

**From Set B (nature-inspired):**
- B1 Slime Mold Pruning (SMNO)
- B2 Immune Clonal Selection (ISCSH)
- B5 Synaptic Pruning (SPDO)
- B6 Ecological Succession (ESO)
- B9 Ant Colony Components (ACPCO)

---

## Set 1: Ideas Informed by Current Experiments (CR/BoK/FSK Findings)

These build on the key insight from our experiments: **context quality matters more than search quantity** (CR matches/beats BoK at half the cost).

---

### 1. Evaluation-Free Screening (EFS) *

**Core idea:** Before spending rollouts evaluating a proposed mutation, ask the reflection LLM itself: "Given this example and this prompt, would the output be correct?" Use the LLM's own prediction as a cheap proxy score. Only spend real rollouts on candidates that pass the screen.

**Why it's novel:** Current methods treat reflection and evaluation as separate phases. EFS fuses them -- the same LLM call that proposes a mutation also predicts its quality. This is analogous to speculative decoding in inference, but applied to prompt optimization.

**Cost:** Each screening call is ~100 tokens (much cheaper than a full task eval with chain-of-thought). If 60% of mutations are screened out, you save ~60% of rollout budget while keeping the good candidates. **~0.6x GEPA rollouts.**

**How it differs from GEPA:** GEPA always evaluates every proposed mutation. EFS adds a cheap filter between proposal and evaluation, skipping obviously bad candidates before they consume rollouts.

**Risk:** LLM self-predictions may not be well-calibrated. Even modest accuracy saves compute, but overconfident screening could reject good candidates.

---

### 2. Active Minibatch Selection (AMS) *

**Core idea:** Instead of epoch-shuffled random minibatches, use information from the Pareto frontier to select maximally informative examples each iteration. Specifically, pick examples where (a) the current best candidate fails, AND (b) at least one other frontier candidate succeeds -- these are the examples where reflection has the most signal.

**Why it's novel:** Standard GEPA treats all training examples as equally informative. AMS exploits the fact that GEPA already tracks per-example scores across all candidates (in `EvaluationCache`). This is essentially active learning applied to the minibatch sampling policy, which hasn't been explored in prompt optimization.

**Cost:** Zero extra LLM calls -- it's a CPU-only change to the `BatchSampler`. The improvement comes from faster convergence (fewer iterations to reach the same quality), meaning fewer total rollouts. **~0.6-0.8x GEPA total cost, same per-iteration cost.**

**How it differs from GEPA:** Replaces `EpochShuffledBatchSampler` with an information-theoretic sampler. Same cost per iteration, fewer iterations needed.

**Risk:** Information-theoretic sampling might overfit to "interesting" examples and miss simpler patterns.

---

### 3. Prompt Delta Optimization (PDO)

**Core idea:** Instead of proposing entire new prompts from scratch, propose small edits (insertions, deletions, rewrites of specific sentences) to the current prompt. The reflection call outputs a structured diff, not a complete rewrite.

**Why it's novel:** All current prompt optimization methods (GEPA, DSPy MIPRO, EvoPrompt) operate on full prompt strings. PDO constrains the mutation space to local edits, which (a) makes the search space much smaller, (b) makes it easier for the reflection LLM to reason about what specifically to change, and (c) enables tracking which edit operations tend to help (building an edit history).

**Cost:** Same as GEPA per iteration, but faster convergence because each mutation is more targeted. Also enables rollback of specific edits that didn't help. **~0.7-0.9x GEPA total cost.**

**How it differs from GEPA:** Changes the granularity of mutation from whole-prompt to sentence-level edits. The reflection prompt becomes "what specific sentence should change and why" rather than "write a whole new system prompt."

**Risk:** May miss improvements that require restructuring the entire prompt.

---

### 4. Contrastive Synthesis Reflection (CSR) *

**Core idea:** Go beyond CR's passive injection. Instead of just showing contrastive pairs in the `<side_info>` section and letting the reflection LLM figure it out, add an explicit synthesis step: a cheap LLM call that analyzes 3-5 contrastive pairs and extracts the abstract principle ("Prompt A works better on math problems because it explicitly asks for step-by-step reasoning, while Prompt B just asks for the answer"). Feed the synthesized principle -- not the raw pairs -- into reflection.

**Why it's novel:** CR shows raw evidence; CSR shows distilled insight. This is the difference between handing someone 5 case studies vs. handing them the conclusion. The synthesis step compresses information and makes it more actionable for the mutation LLM.

**Cost:** One additional cheap LLM call per iteration (short input, short output -- maybe 500 tokens total). But the improved reflection quality should reduce the number of iterations needed. **~0.8-0.9x GEPA total cost.**

**How it differs from CR:** CR injects raw contrastive pairs. CSR distills them into principles first. Think of it as "CR with a reasoning layer."

**Risk:** The synthesis step might over-simplify or misinterpret the contrastive evidence.

---

### 5. Progressive Model Cascade (PMC)

**Core idea:** Run prompt optimization in two phases. Phase 1: use a cheap/fast model (e.g., Qwen-7B or even the task model itself) as the reflection LLM for broad exploration -- many iterations cheaply. Phase 2: switch to a strong model (Qwen-27B or Claude) for refinement, starting from Phase 1's best candidate. Allocate 70% of budget to Phase 1, 30% to Phase 2.

**Why it's novel:** All current methods use a single reflection model throughout. PMC applies the curriculum learning insight -- cheap models can identify promising directions even if they can't write perfect prompts. The expensive model then polishes the promising directions.

**Cost:** Phase 1 is ~4x cheaper per iteration (7B vs 27B). If Phase 1 handles 70% of iterations, total cost drops by ~50% while final quality remains similar. **~0.5x GEPA reflection cost.**

**How it differs from GEPA:** Replaces the fixed reflection LM with a two-stage cascade. Same task evaluator throughout (so rollout costs are unchanged), but reflection costs drop significantly.

**Risk:** Cheap model's exploration might lead to dead ends that the expensive model can't recover from.

---

### 6. Failure-Clustered Targeted Mutation (FCTM) *

**Core idea:** Before each mutation cycle, cluster the failing examples by failure type (wrong format, wrong reasoning, partial credit, etc.) using embedding similarity on the LLM's outputs. Generate one targeted mutation per failure cluster (e.g., "fix the formatting failures" separately from "fix the reasoning failures"). This naturally produces diverse mutations without the overhead of BoK's K-sampling.

**Why it's novel:** FSK partitions failures randomly (round-robin/worst-first). FCTM partitions them semantically. This means each mutation addresses a coherent failure mode rather than an arbitrary subset. The mutations are diverse because the failure modes are different, not because we generate K random variants.

**Cost:** One embedding call to cluster failures (cheap, can use a small embedding model), then N mutation calls where N = number of clusters (typically 2-4). Comparable to BoK but with much more targeted mutations. **~1.5x GEPA per iteration (but fewer iterations needed).**

**How it differs from FSK:** FSK assigns failures arbitrarily to K buckets. FCTM discovers natural failure clusters. This should produce better mutations because each one addresses a coherent problem.

**Risk:** Clustering quality depends on the embedding model. Poor clusters = poor mutations.

---

### 7. Reflection Trajectory Caching (RTC) *

**Core idea:** Cache the reflection patterns -- not just the evaluation scores. When the reflection LLM sees a failure pattern it has seen before (e.g., "model outputs raw number instead of boxed answer"), reuse the previous mutation strategy instead of generating a new reflection. Detect pattern similarity using a hash of the failure type (extractable from trajectories).

**Why it's novel:** GEPA treats every iteration's reflection as independent. In practice, the same failure patterns recur across iterations (especially early on when the prompt is bad in systematic ways). RTC amortizes the reflection cost by reusing successful mutation strategies for recurring problems.

**Cost:** Dramatically lower -- reflection calls drop to zero for repeated failure patterns. The first time a pattern is seen, full reflection runs. After that, cached. Over a 7000-rollout run, this could cut reflection calls by 50-70%. **~0.3-0.5x GEPA reflection cost, same rollout cost.**

**How it differs from GEPA:** Adds a cache layer before the reflection LLM. Same quality (the cached mutations were already validated), much lower cost.

**Risk:** Cached strategies may become stale as the prompt evolves (a fix that worked at iteration 10 might not apply at iteration 500).

---

### 8. Pareto-Guided Crossover (PGC)

**Core idea:** Instead of mutating a single parent candidate, perform structured crossover between two Pareto-frontier candidates that have complementary strengths -- i.e., Candidate A excels on examples where B fails, and vice versa. The reflection call receives both candidates and their performance profiles, and is asked to synthesize a child that combines their strengths.

**Why it's novel:** GEPA's merge proposer exists but operates differently -- it merges candidates that are close in score space. PGC explicitly selects candidates with complementary failure profiles, which is more likely to produce genuine improvement. This is inspired by genetic algorithm crossover but guided by the Pareto frontier's diversity.

**Cost:** Same as GEPA per iteration (one reflection call, one evaluation). The improvement comes from better parent selection, not more compute. **~1.0x GEPA per iteration, fewer iterations to converge.**

**How it differs from GEPA:** GEPA mutates one parent. PGC synthesizes from two complementary parents. Uses the frontier's diversity as a feature rather than just a selection mechanism.

**Risk:** Combining two prompts may produce incoherent results if the strengths aren't compositional.

---

### 9. Implicit Optimization via Example Curation (IOEC)

**Core idea:** Instead of optimizing the system prompt directly, optimize the set of few-shot examples included in the prompt. Start with no examples, then iteratively: (a) find training examples the current prompt fails on, (b) generate a correct solution for each, (c) add the best (example, solution) pair to the prompt as a few-shot demonstration. The "mutation" is adding/removing/replacing a few-shot example rather than rewriting instructions.

**Why it's novel:** Most prompt optimization focuses on instruction text. IOEC optimizes the demonstration component, which is often more impactful for complex tasks (especially reasoning-heavy ones like AIME). Few-shot example selection has been studied, but iterative curation with failure-guided selection hasn't been combined with GEPA's framework.

**Cost:** Much cheaper per iteration -- no reflection LLM call needed. The "mutation" is just selecting which example to add/remove, which can be done with heuristics on the evaluation scores. Only cost is the task evaluation. **~0.5x GEPA per iteration (no reflection call).**

**How it differs from GEPA:** Changes what gets optimized (examples vs. instructions). Can also be combined with GEPA as a hybrid (optimize both instructions and examples jointly).

**Risk:** Some tasks don't benefit from few-shot examples (or the prompt gets too long).

---

### 10. Self-Consistency Guided Optimization (SCGO)

**Core idea:** Use self-consistency (run the task model multiple times on each example and check agreement) as a free signal for identifying which examples the current prompt is fragile on. Examples with low self-consistency (model gives different answers across runs) are the most improvable -- small prompt changes can flip them. Focus reflection on these fragile examples rather than hard failures.

**Why it's novel:** Standard GEPA uses a single evaluation per example and focuses on failures. But failures come in two types: (a) hard failures (the model fundamentally can't solve this) and (b) fragile examples (the model sometimes gets it right). Optimizing for (b) has much higher ROI because a prompt tweak can reliably flip fragile examples to correct.

**Cost:** Requires K task evals per example (K=3-5) on the initial pass to estimate consistency, but this cost is amortized -- once you know which examples are fragile, you can focus all subsequent iterations on them. Over a full run, total rollouts may decrease because you converge faster. **~0.8-1.2x GEPA total, depending on how many fragile examples exist.**

**How it differs from GEPA:** Adds a consistency-based example importance signal. Changes which examples get attention during reflection, leading to more impactful mutations.

**Risk:** Initial consistency estimation is expensive (3-5x for first pass). May not help if most failures are "hard" rather than "fragile."

---

### Set 1 Top 5 (excluding #5 PMC, per user request)

| Rank | Idea | Why |
|------|------|-----|
| 1 | **#2 Active Minibatch Selection** | Zero extra cost, purely smarter data selection. Infrastructure already exists (EvaluationCache). Easiest to implement. |
| 2 | **#4 Contrastive Synthesis Reflection** | Natural evolution of CR (our best finding). One cheap extra call distills raw pairs into principles. |
| 3 | **#7 Reflection Trajectory Caching** | Biggest cost savings (50-70% fewer reflection calls). Same failure patterns genuinely recur. |
| 4 | **#1 Evaluation-Free Screening** | 40% rollout savings. Composable with every other method. |
| 5 | **#6 Failure-Clustered Targeted Mutation** | Semantic upgrade over FSK. Makes diversity meaningful, not random. |

---

## Set A: Novel Ideas (Independent of Current Experiments)

These don't build on GEPA, CR, BoK, or FSK at all -- they're fundamentally different approaches to the prompt optimization problem.

---

### A1. Zero-Shot Prompt Synthesis (ZPS)

**Core idea:** Skip iterative optimization entirely. In a single LLM call, provide: (a) 10 diverse training examples spanning difficulty levels, (b) the evaluation metric definition, (c) 5 failed outputs from a blank/naive prompt with error analysis. Ask the LLM to write the optimal prompt in one shot, given full context about what the task needs and what goes wrong without a good prompt.

**Why it could beat GEPA:** GEPA spends thousands of rollouts discovering things the LLM could infer from a handful of examples. A strong reflection model (Claude/GPT-4) can often write a near-optimal prompt if given enough context about the task structure. ZPS frontloads all the context instead of discovering it iteratively.

**Cost:** 1 reflection call + ~15 task evaluations (5 for failure analysis, 10 for final scoring). **~0.2% of GEPA's rollout budget.** Even if it only reaches 75% of GEPA's quality, the cost-quality tradeoff is extraordinary.

**Risk:** May hit a quality ceiling on complex tasks where the failure modes aren't obvious from 5 examples. But it could serve as initialization for any iterative method -- ZPS + 500 rollouts of GEPA might beat 7000 rollouts of GEPA from a blank seed.

---

### A2. Surrogate-Guided Bayesian Prompt Optimization (SBPO)

**Core idea:** Embed each candidate prompt into a continuous vector (using a text embedding model like E5 or GTE). After evaluating ~50 prompts, train a Gaussian Process (GP) regressor mapping embedding vectors to scores. Use the GP's acquisition function (Expected Improvement or UCB) to propose the region of embedding space most likely to contain a high-scoring prompt. Decode that region back to text by asking the LLM: "Write a prompt whose meaning is similar to [nearest evaluated prompt] but shifted toward [direction of improvement]."

**Why it could beat GEPA:** GEPA searches in text space, which is combinatorially enormous and discrete. SBPO searches in a low-dimensional continuous space where Bayesian optimization is provably efficient. The GP provides uncertainty estimates -- it knows where it hasn't explored, enabling principled exploration-exploitation tradeoff rather than GEPA's heuristic Pareto selection.

**Cost:** ~50-100 task evaluations to train the surrogate + ~20-30 more guided by the GP. **Total ~80-130 evaluations vs GEPA's thousands (~1-2%).** The GP inference and embedding calls are negligible.

**Risk:** The embedding space may not preserve the structure that matters for prompt quality (semantic similarity != functional similarity). Mitigation: use a task-specific embedding or learn the embedding jointly.

---

### A3. Prompt Tournament Selection (PTS) *

**Core idea:** Generate a large, diverse pool of 64 candidate prompts in a single batched LLM call (ask for 64 different approaches to the task). Run a single-elimination tournament: randomly pair prompts, evaluate each pair on a small shared subset (5 examples), advance the winner. After 6 rounds (log2(64)), you have the champion. Optionally, run a "consolation bracket" to identify 2nd and 3rd place for ensemble or merge.

**Why it could beat GEPA:** GEPA evaluates sequentially (propose one, evaluate, propose another). PTS evaluates comparatively -- you only need to determine which of two prompts is better, not their absolute scores. Comparative evaluation on 5 examples is much more reliable than absolute scoring because the noise cancels out (same 5 examples for both). Total evaluations: 63 matchups x 5 examples x 2 prompts = 630 rollouts.

**Cost:** 1 batch LLM call for generation + 630 task evaluations. **~9% of GEPA's budget** for a 7051-rollout benchmark. Much faster wall-clock because matchups within a round can run in parallel.

**Risk:** The initial pool quality matters -- if no good prompt is in the starting 64, the tournament can't find one. Mitigation: use diverse generation strategies (chain-of-thought, few-shot, structured, etc.) and allow a "wildcard" round where top-8 losers get one mutation chance.

---

### A4. Adversarial Minimax Prompt Optimization (AMPO) *

**Core idea:** Instead of maximizing average score (what GEPA does), minimize worst-case failure. Two-player game: an "attacker" LLM identifies the hardest examples for the current prompt (finds inputs most likely to cause failure), and a "defender" LLM strengthens the prompt specifically against those attacks. Iterate until the attacker can't find failures.

**Why it could beat GEPA:** GEPA optimizes for average performance, which can leave systematic blind spots (the prompt might score 90% average but consistently fail on a specific example type). AMPO explicitly hunts for and eliminates blind spots. In practice, worst-case optimization often improves average-case too, because fixing systematic failures lifts the floor.

**Cost:** Each iteration: 1 attack call (cheap -- just find hard examples) + 1 defense call (reflection) + ~N task evaluations. Similar per-iteration cost to GEPA, but converges faster because each iteration addresses the most impactful failure. **~0.8x GEPA total.**

**Risk:** The attacker might find "impossible" examples that no prompt can solve, wasting defender iterations. Mitigation: only surface examples where at least one known prompt succeeds (proves solvability).

---

### A5. Prompt Decomposition and Modular Optimization (PDMO) *

**Core idea:** Instead of optimizing a monolithic prompt, decompose it into independent modules: (1) task framing ("You are solving math problems"), (2) reasoning strategy ("Think step by step"), (3) format constraints ("Box your final answer"), (4) error prevention ("Double-check your arithmetic"). Optimize each module independently on a small budget (~500 rollouts each). Compose the best modules into the final prompt.

**Why it could beat GEPA:** The search space for a full prompt is the product of all module spaces. By decomposing, you search each factor independently -- the total cost is the sum of factor costs, not the product. For 4 modules with 500-rollout budgets each, total cost is 2000 rollouts vs GEPA's 7000. And each module optimization converges faster because the search space is smaller.

**Cost:** K x 500 rollouts where K = number of modules (typically 3-5). Plus ~100 rollouts for final composition testing. **Total: ~1600-2600 rollouts (~30-40% of GEPA).**

**Risk:** Modules may interact -- the best reasoning strategy might depend on the task framing. Mitigation: after independent optimization, run a short (500-rollout) joint refinement phase on the composed prompt. Still much cheaper than monolithic optimization.

---

### A6. Multi-Benchmark Meta-Prompt Template (MBMT)

**Core idea:** Instead of optimizing 6 separate prompts for 6 benchmarks, optimize a single meta-template with benchmark-specific slots:

```
You are an expert at {TASK_TYPE}. {DOMAIN_CONTEXT}
When solving problems: {REASONING_STRATEGY}
{FORMAT_INSTRUCTIONS}
{ERROR_PREVENTION}
```

Optimize the template structure and shared components across all benchmarks simultaneously. Only the slot contents are benchmark-specific (and much shorter, so cheaper to optimize individually).

**Why it could beat GEPA:** Amortizes optimization cost. Instead of 6 x 7000 = 42,000 rollouts, you spend ~5000 rollouts on the shared template + 6 x 500 = 3000 rollouts on slot customization = 8000 total. **5x cheaper across all benchmarks.** Also enables transfer learning -- what works for aime's reasoning strategy might help livebench.

**Cost:** ~8000 total rollouts across all benchmarks vs 42,000 for independent GEPA. **~20% amortized cost.**

**Risk:** Some benchmarks may need fundamentally different prompt structures. Mitigation: allow the template itself to have optional sections that can be toggled per benchmark.

---

### A7. Cooperative Heterogeneous Search Ensemble (CHSE)

**Core idea:** Run 5 fundamentally different optimization strategies in parallel, each allocated 1/5 of the total rollout budget:
1. Random search (generate and evaluate many diverse prompts)
2. GEPA-style reflective mutation
3. Template-based combinatorial search
4. Example-driven synthesis (ZPS-style, repeated with different example subsets)
5. Compression-based (start long, prune while maintaining quality)

Every 200 rollouts, all agents share their current best prompt. Each agent can adopt another's best as its starting point if it's better than its own.

**Why it could beat GEPA:** Different strategies excel in different regions of the search space. Random search finds good starting points, reflection refines them, compression makes them efficient. The sharing mechanism means the ensemble converges to the globally best prompt regardless of which strategy found it. No single strategy needs to be optimal -- the ensemble is.

**Cost:** Same total budget as GEPA, but uses it more efficiently by covering more of the search space. **Wall-clock is 5x faster if run in parallel.** Same total cost, better result.

**Risk:** Coordination overhead and strategy interference. Mitigation: keep sharing infrequent (every 200 rollouts) and one-directional (only share if your best beats the global best).

---

### A8. Constraint-First Principle Optimization (CFPO) *

**Core idea:** Instead of optimizing prompt text directly, optimize a set of abstract principles that the prompt must embody. Example principles: "always show intermediate steps", "restate the question before answering", "consider edge cases explicitly". The optimization searches over principle combinations (a much smaller discrete space -- maybe 20 candidate principles, choose 5 = C(20,5) = 15,504 combinations). For each principle combination, an LLM renders it into prompt text (cheap, deterministic). Only evaluate the rendered prompts.

**Why it could beat GEPA:** The principle space is combinatorial but small (15K combinations vs infinite prompt text space). Exhaustive search is feasible with smart pruning. Each principle can be validated independently first (does adding "show steps" help on its own?), reducing the effective search space to combinations of individually useful principles.

**Cost:** ~20 individual principle evaluations (x N examples each) + ~50-100 combination evaluations. **Total: ~500-2000 rollouts (~7-30% of GEPA).**

**Risk:** The principle-to-prompt rendering may lose nuance. Mitigation: include a short refinement phase after finding the best principle set, using GEPA-style reflection on the rendered prompt (budget ~500 rollouts).

---

### A9. Score-Predictive Prompt Ranking (SPPR)

**Core idea:** Train a lightweight neural network (a small transformer or MLP on top of prompt embeddings) to predict the score of a prompt on a benchmark. Training data: evaluate 100-200 diverse prompts to get (prompt, score) pairs. Once trained, use the network to score thousands of candidate prompts instantly (no task evaluation). Only run real evaluations on the top-10 predictions.

**Why it could beat GEPA:** After the initial 200-evaluation training phase, candidate scoring is essentially free. You can generate and score 10,000 prompts in seconds. The quality bottleneck shifts from evaluation cost to generation diversity -- and diverse generation is cheap (one batched LLM call).

**Cost:** 200 initial evaluations + 10 verification evaluations = 210 total. **~3% of GEPA's budget.** The neural network training and inference is negligible (CPU, <1 minute).

**Risk:** The predictor might not generalize to prompt structures very different from its training data. Mitigation: active learning -- when the predictor is uncertain, evaluate that prompt and add it to training data.

---

### A10. Dual Critic-Guided Optimization (DCGO) *

**Core idea:** Optimize two prompts simultaneously: a "solver" prompt (the one being optimized) and a "critic" prompt. The critic prompt takes a task input + the solver's output and predicts whether the output is correct -- it's a learned verifier. The critic is much cheaper to run than re-evaluating with ground truth (no chain-of-thought needed, just classification). Use the critic to score hundreds of solver prompt variants cheaply, then verify only the top candidates with real evaluation.

**Why it could beat GEPA:** The critic acts as a cheap surrogate evaluator. After training the critic on ~100 examples with ground truth, you can "evaluate" new solver prompts at ~10x lower cost (the critic call is short and doesn't need chain-of-thought). This is similar to SPPR but uses an LLM-based critic instead of a neural network -- potentially more accurate.

**Cost:** 100 evaluations to train/calibrate the critic + ~50 critic-scored iterations + ~20 real evaluations to verify top candidates = ~170 real evaluations total. **~2.5% of GEPA's budget.**

**Risk:** Critic might develop blind spots (consistently approve a specific error type). Mitigation: periodically recalibrate the critic with fresh real evaluations.

---

## Set B: Nature-Inspired Ideas

These draw inspiration from biological and natural optimization processes.

---

### B1. Slime Mold Network Optimization (SMNO) *

**Inspired by:** *Physarum polycephalum* -- the slime mold that finds shortest paths in mazes by growing along all possible paths simultaneously, then pruning inefficient branches as nutrients flow preferentially through shorter paths.

**Core idea:** Generate 20 diverse prompts (growth phase). Evaluate all 20 on a small subset (nutrient distribution). "Flow" is proportional to score -- high-scoring prompts get more "nutrients" (additional mutation attempts). Low-scoring prompts are starved (pruned). After one round: 20 -> 10. Mutate the survivors, evaluate again. 10 -> 5. Continue until 1 champion remains.

**Why it could beat GEPA:** GEPA commits to a single mutation path and backtracks via the Pareto frontier. SMNO explores many paths simultaneously and prunes based on flow (score). The parallel exploration covers more of the search space, and the progressive pruning focuses resources on the most promising regions.

**Cost:** Round 1: 20 x 10 = 200 rollouts. Round 2: 10 x 15 = 150. Round 3: 5 x 20 = 100. Round 4: 3 x 30 = 90. **Total: ~540 rollouts (~8% of GEPA).**

**Risk:** Initial diversity quality matters. If the 20 starting prompts don't cover the good regions, pruning won't help.

---

### B2. Immune System Clonal Selection and Hypermutation (ISCSH) *

**Inspired by:** Adaptive immune response -- the body generates billions of random antibodies, identifies which ones bind to a pathogen, then clones the winners and hypermutates the clones at a rate 10,000x higher than normal. The best-binding mutants survive; the rest die.

**Core idea:** Phase 1 (generation): Generate 50 diverse prompt "antibodies" cheaply. Phase 2 (screening): Evaluate each on 3 examples -- a coarse affinity test. Phase 3 (clonal selection): Take the top 5, generate 10 aggressive mutations of each (50 mutants total) using high-temperature LLM sampling. Phase 4 (affinity maturation): Evaluate all 50 mutants on 20 examples, keep the top 3. Phase 5: Final evaluation of top 3 on full validation set.

**Why it could beat GEPA:** The immune system is optimized for finding a needle in a haystack with minimal resources. The key insight is variable mutation rate: early exploration uses wild, diverse mutations (high temperature), while later refinement uses conservative mutations (low temperature). GEPA uses a fixed mutation strategy throughout.

**Cost:** 50x3 + 50x20 + 3xfull_val = 150 + 1000 + ~300 = **~1450 rollouts (~20% of GEPA).**

**Risk:** Coarse screening (3 examples) might eliminate good candidates that happen to fail on those specific 3 examples. Mitigation: use stratified example selection for screening.

---

### B3. Mycorrhizal Knowledge Network (MKN)

**Inspired by:** Mycorrhizal networks -- underground fungal networks connecting trees in a forest. Larger, healthier trees share carbon and nutrients with smaller, struggling trees through the fungal network. The sharing is asymmetric (strong to weak) and happens through a shared medium.

**Core idea:** Run N=5 independent GEPA-like optimization streams, each with 1/5 the budget. Maintain a shared "knowledge network" (a database of prompt components + their scores on specific example types). After each iteration, every stream deposits its discoveries (which prompt components worked for which examples). Struggling streams (low scores) can pull from the network -- accessing components that worked for similar examples in other streams. High-performing streams continue independently.

**Why it could beat GEPA:** Single GEPA wastes rollouts rediscovering things. With 5 streams sharing knowledge, each discovery benefits all streams. The asymmetric sharing (struggling streams pull from successful ones) prevents the network from being dominated by one strategy.

**Cost:** Same total budget as GEPA. Each stream gets budget/5. But the shared network makes each stream more efficient, so the effective budget is >budget. **1.0x total cost, ~2-3x effective search coverage.**

**Risk:** The knowledge network must be structured carefully -- raw prompt sharing without context could introduce noise.

---

### B4. Metamorphosis-Based Prompt Restructuring (MBPR)

**Inspired by:** Insect metamorphosis -- the caterpillar completely dissolves its body inside the chrysalis, then rebuilds as a butterfly using the same raw materials but a fundamentally different architecture.

**Core idea:** Phase 1 (larval/caterpillar): Run standard GEPA-style optimization until convergence or plateau (~2000 rollouts). Phase 2 (chrysalis): "Dissolve" the best prompt -- ask an LLM to extract the abstract principles, strategies, and insights from the prompt WITHOUT preserving its structure. Output: a list of 5-10 principles. Phase 3 (butterfly): Rebuild the prompt from scratch using these principles but with a completely different structure (different ordering, different phrasing, different level of detail). Evaluate the new structure. Phase 4: Short refinement (~500 rollouts).

**Why it could beat GEPA:** GEPA gets trapped by its prompt structure -- iterative mutation preserves the skeleton while tweaking details. Metamorphosis allows a complete structural reset while preserving the knowledge accumulated during optimization. This breaks out of structural local optima. The rebuilt prompt often outperforms the original because the dissolution step forces distillation of what actually matters.

**Cost:** ~2000 + 1 LLM call + 1 LLM call + ~500 = **~2500 rollouts (~35% of GEPA).**

**Risk:** The dissolution step might lose important nuances. Mitigation: keep the original prompt as a fallback and only accept the restructured version if it scores higher.

---

### B5. Synaptic Pruning via Developmental Optimization (SPDO) *

**Inspired by:** Brain development -- infants are born with far more synaptic connections than adults. During childhood, unused connections are pruned while heavily-used connections are strengthened. The brain becomes more efficient by removing structure, not adding it.

**Core idea:** Start with an extremely detailed, over-specified prompt (~2000 words covering every possible edge case, format requirement, reasoning strategy, and error prevention). Evaluate it (should score well due to thoroughness). Then iteratively prune: remove one section/sentence at a time, re-evaluate. If the score doesn't drop (or drops <1%), the section was "unused" -- permanently remove it. If the score drops significantly, that section is "load-bearing" -- keep it and strengthen it (add detail). Continue until the prompt is minimal but retains full performance.

**Why it could beat GEPA:** GEPA builds up from a minimal seed, which risks missing important components. SPDO starts with "everything" and removes what's unnecessary -- guaranteed to not miss any important component. The final prompt is also more efficient at inference time (shorter prompts = cheaper/faster task model calls). The initial over-specified prompt is easy to generate (just ask the LLM to be extremely thorough).

**Cost:** 1 LLM call to generate the over-specified prompt. Then ~30-50 pruning evaluations (each tests removing one component). **Total: ~200-500 rollouts (~3-7% of GEPA).**

**Risk:** The initial over-specified prompt might be so verbose that it confuses the model. Mitigation: generate several versions and pick the one that scores best before starting pruning.

---

### B6. Ecological Succession Optimization (ESO) *

**Inspired by:** Ecological succession -- bare rock -> lichens -> mosses -> grasses -> shrubs -> forest. Each stage prepares the environment for the next. Pioneer species (simple, hardy) establish first; climax species (complex, specialized) come later.

**Core idea:** Optimize in stages, each building on the previous:
- Stage 1 (pioneer): Optimize on the 20% easiest examples only (small, fast). Budget: 500 rollouts.
- Stage 2 (shrub): Optimize on the 50% easiest examples, starting from Stage 1's best prompt. Budget: 1000 rollouts.
- Stage 3 (forest/climax): Optimize on all examples, starting from Stage 2's best prompt. Budget: 2000 rollouts.

**Why it could beat GEPA:** GEPA throws all examples at the prompt from the start, including impossible ones that waste rollouts. ESO builds competence progressively -- solving easy problems first establishes the fundamental strategies, which then transfer to harder problems. This is curriculum learning applied to prompt optimization.

**Cost:** 500 + 1000 + 2000 = **3500 rollouts (~50% of GEPA).** And the curriculum structure means each rollout is more informative because the prompt-to-difficulty match is better.

**Risk:** Easy-first optimization might learn strategies that don't generalize to hard problems. Mitigation: at each stage transition, allow the optimizer to restructure (not just refine) the prompt.

---

### B7. Quorum Sensing Convergence Detection (QSCD)

**Inspired by:** Bacterial quorum sensing -- bacteria release signaling molecules. When the local concentration exceeds a threshold (indicating sufficient population density), all bacteria simultaneously switch behavior (e.g., form a biofilm). Individual bacteria can't determine the right time to switch, but the collective signal is reliable.

**Core idea:** Run 7 independent prompt optimizations, each with a different seed, different initialization, and 1/7 the budget. After every 100 rollouts, each optimizer shares its current best prompt. When 4+ of the 7 converge to structurally similar prompts (measured by sentence-level overlap or embedding similarity), that's a "quorum" -- those shared structural elements are robust and not seed-dependent noise. Lock in the consensus elements and let all optimizers refine the remaining non-consensus parts.

**Why it could beat GEPA:** Single GEPA can't distinguish robust improvements from lucky noise on a specific minibatch. QSCD uses cross-seed convergence as a natural noise filter. If 5/7 optimizers independently discover that "think step by step" helps, that's strong evidence. If only 1/7 includes "consider the problem from multiple angles," it's probably noise. This is ensembling applied to the optimization process, not the final prediction.

**Cost:** Same total budget as GEPA (7 x budget/7). But the quorum mechanism accelerates convergence by filtering noise early. **1.0x total cost, 2-3x faster effective convergence.**

**Risk:** If all 7 optimizers converge prematurely to a local optimum, quorum sensing would lock that in. Mitigation: require quorum only for structural elements, not full prompt text.

---

### B8. Coral Reef Niche Specialization (CRNS)

**Inspired by:** Coral reef ecosystems -- extreme biodiversity in a small area because each species occupies a specific niche. No single species dominates; the ecosystem's strength comes from specialization and complementarity.

**Core idea:** Instead of finding one general-purpose prompt, evolve a portfolio of 3-5 specialist prompts, each adapted to a "niche" of examples. Identify niches by clustering training examples by type (using embeddings). Optimize a specialist prompt for each cluster with a small budget (budget/5 each). At inference time, classify each test example into a niche and route it to the corresponding specialist.

**Why it could beat GEPA:** One-size-fits-all prompts make compromises. A prompt optimized for step-by-step math reasoning might hurt on problems that need creative insight. CRNS eliminates this tradeoff. Each specialist can be fully optimized for its niche without compromising on others. The routing classifier is cheap (embedding similarity, no LLM call).

**Cost:** Clustering: 1 embedding call. 5 specialist optimizations: 5 x budget/5 = same total budget. But each specialist converges faster (smaller, more homogeneous training set), so effective cost may be **60-70% of GEPA.** Inference cost: 1 embedding call per example + the task eval (same).

**Risk:** Test examples might not fit neatly into clusters. Mitigation: include a "generalist" prompt for examples that don't clearly belong to any niche.

---

### B9. Ant Colony Prompt Component Optimization (ACPCO) *

**Inspired by:** Ant colony optimization -- ants find shortest paths to food by depositing pheromones. More-traveled paths accumulate more pheromone and attract more ants (positive feedback). Pheromone evaporates over time (forgetting mechanism), so abandoned paths are eventually forgotten.

**Core idea:** Decompose prompts into a library of ~50 atomic components (individual sentences/instructions). Each component has a "pheromone level" (initially uniform). To construct a candidate prompt, sample 8-12 components proportional to their pheromone levels. Evaluate the constructed prompt. Update pheromone: increase for all components in prompts that scored above-median, decrease (evaporate) for components in below-median prompts. Over time, high-pheromone components naturally aggregate into the best prompt.

**Why it could beat GEPA:** GEPA treats the prompt as an opaque string. ACPCO treats it as a composition of independent components with measurable value. This enables: (a) credit assignment -- which specific sentence caused the improvement, (b) transfer -- good components discovered in one prompt benefit future prompts, (c) convergence detection -- when pheromone levels stabilize, you've found the optimal composition.

**Cost:** ~200-500 constructed prompts evaluated. Each is cheap (pheromone sampling is instant). **Total: ~1000-2500 rollouts (~15-35% of GEPA).** The pheromone mechanism provides strong guidance after ~50 evaluations.

**Risk:** Components may interact in ways the pheromone model can't capture. Mitigation: include bigram pheromone (pairs of components that work well together) in addition to unigram pheromone.

---

### B10. Predator-Prey Co-evolutionary Optimization (PPCO)

**Inspired by:** Arms races in predator-prey evolution -- cheetahs evolve speed, gazelles evolve agility, cheetahs evolve better vision, gazelles evolve better camouflage. Both species improve because the pressure from each other is always exactly calibrated -- neither too easy nor too hard.

**Core idea:** Evolve two populations simultaneously: (1) "prey" = prompt candidates trying to maximize score, (2) "predators" = adversarial example selectors trying to find examples that defeat the best prompts. The predator population evolves to present increasingly difficult, targeted challenges. The prey population evolves to handle increasingly sophisticated attacks. The co-evolutionary pressure ensures that prompts improve in the most impactful direction at every step.

**Why it could beat GEPA:** GEPA evaluates on random minibatches -- many examples are "too easy" (already solved) or "too hard" (unsolvable), wasting rollouts. PPCO automatically calibrates difficulty -- the predator population ensures the prey always faces challenges at the edge of its capability, maximizing the information content of every evaluation.

**Cost:** Comparable per-iteration cost to GEPA (one mutation + one evaluation), but each evaluation is maximally informative due to adversarial example selection. **Convergence in ~50-70% of GEPA's iterations.**

**Risk:** Co-evolutionary dynamics can be unstable (Red Queen effect -- both populations cycle without progress). Mitigation: archive the best prompt from each generation to prevent regression.

---

## Master Summary Table

| # | Name | Cost vs GEPA | Key Mechanism |
|---|------|-------------|---------------|
| **1** | Evaluation-Free Screening | ~60% | Skip bad mutations before evaluating |
| **2** | Active Minibatch Selection | 60-80% | Pick maximally informative examples |
| **3** | Prompt Delta Optimization | 70-90% | Edit sentences, not whole prompts |
| **4** | Contrastive Synthesis Reflection | 80-90% | Distill contrastive pairs into principles |
| **5** | Progressive Model Cascade | ~50% refl | Cheap model explores, expensive refines |
| **6** | Failure-Clustered Targeted Mutation | ~150%/iter, fewer iters | Semantic failure clustering > random partition |
| **7** | Reflection Trajectory Caching | 30-50% refl | Reuse mutation strategies for recurring failures |
| **8** | Pareto-Guided Crossover | 100%/iter, fewer iters | Synthesize from complementary parents |
| **9** | Implicit Optimization via Examples | ~50% | Optimize few-shot examples, not instructions |
| **10** | Self-Consistency Guided | 80-120% | Focus on fragile examples, not hard failures |
| **A1** | Zero-Shot Synthesis | 0.2% | One-shot generation with full context |
| **A2** | Bayesian Surrogate | 1-2% | GP in embedding space |
| **A3** | Tournament Selection | 9% | Single-elimination bracket |
| **A4** | Adversarial Minimax | ~80% | Worst-case optimization |
| **A5** | Modular Decomposition | 30-40% | Independent module optimization |
| **A6** | Meta-Prompt Template | ~20% amortized | Shared template + slots |
| **A7** | Heterogeneous Ensemble | 100% (5x faster wall) | Diverse parallel strategies |
| **A8** | Constraint/Principle Optimization | 7-30% | Optimize principles, not text |
| **A9** | Score-Predictive Network | 3% | Learned surrogate scorer |
| **A10** | Dual Critic-Guided | 2.5% | LLM verifier as cheap evaluator |
| **B1** | Slime Mold Pruning | 8% | Parallel exploration + progressive pruning |
| **B2** | Immune Clonal Selection | 20% | Variable mutation rate + clone winners |
| **B3** | Mycorrhizal Network | 100% (2-3x coverage) | Shared knowledge across streams |
| **B4** | Metamorphosis | 35% | Dissolve + rebuild from principles |
| **B5** | Synaptic Pruning | 3-7% | Start maximal, prune non-contributing parts |
| **B6** | Ecological Succession | 50% | Easy->medium->hard curriculum |
| **B7** | Quorum Sensing | 100% (2-3x faster) | Cross-seed convergence as noise filter |
| **B8** | Coral Reef Niches | 60-70% | Specialist prompts per example cluster |
| **B9** | Ant Colony Components | 15-35% | Pheromone-weighted component composition |
| **B10** | Predator-Prey | 50-70% | Co-evolved adversarial example selection |

---

## Final Top 5: Non-Nature Ideas to Implement (Single-Model Constraint)

From the 10 non-nature top picks (Set 1 + Set A), these 5 form the strongest portfolio for beating GEPA with a single model:

| Rank | Idea | Cost | Rationale |
|------|------|------|-----------|
| 1 | **#2 Active Minibatch Selection** | 60-80% | Zero extra cost, composable with everything. Just a smarter BatchSampler using data already in EvaluationCache. Easiest to implement, guaranteed not to hurt. |
| 2 | **A5 Modular Decomposition** | 30-40% | Most principled cost reduction. Searching 4 small independent spaces is provably cheaper than 1 large space. Short joint refinement phase handles module interactions. |
| 3 | **A3 Tournament Selection** | 9% | Radically different paradigm. 64 diverse prompts competing head-to-head. Comparative evaluation on shared examples is statistically more powerful than absolute scoring. Can seed GEPA for a final refinement push if needed. |
| 4 | **#4 Contrastive Synthesis Reflection** | 80-90% | Builds directly on CR (our strongest experimental finding). One cheap synthesis call distills contrastive pairs into actionable principles. Minimal implementation effort, high expected payoff. |
| 5 | **#1 Evaluation-Free Screening** | ~60% | Composable cost multiplier -- stacks on top of any other method. LLM self-prediction filters bad mutations before they consume rollouts. Even conservative screening (reject only high-confidence "no") saves 30%+ rollouts. |

**Why these 5 together:** They cover orthogonal improvement axes:
- **AMS** improves *what data* the optimizer sees (smarter sampling)
- **PDMO** improves *where* it searches (smaller, decomposed spaces)
- **PTS** is a *completely different search paradigm* (comparative, not iterative)
- **CSR** improves *what context* the reflection LLM gets (distilled insights)
- **EFS** reduces *wasted compute* (skip bad candidates before evaluating)

And critically, AMS + CSR + EFS are composable -- they can stack on top of GEPA simultaneously for a combined ~30-40% cost reduction with improved quality. PDMO and PTS are standalone alternatives that could beat GEPA outright at a fraction of the cost.

**What was cut and why:**
- **RTC (#7):** Promising but cache invalidation is tricky -- cached strategies go stale as the prompt evolves. Higher implementation risk.
- **FCTM (#6):** 1.5x per-iteration cost is a hard sell when we're optimizing for cheaper. Needs an embedding model for clustering, adding infrastructure complexity.
- **CFPO (A8):** The principle-to-prompt rendering step is a black box -- if it loses nuance, the whole approach fails. Less predictable than the selected 5.
- **DCGO (A10):** Critic calibration is fragile. Blind spots in the critic silently corrupt optimization. Hard to debug.
- **AMPO (A4):** ~80% of GEPA cost isn't cheap enough. Best for quality ceiling, not cost reduction.

---

## The 8 Methods We Will Test

These are the final 8 methods selected for experimentation, organized by how they relate to GEPA:

### Stack on GEPA (2) -- augment GEPA's existing loop

| Method | Cost vs GEPA | What it improves |
|--------|-------------|------------------|
| **#2 Active Minibatch Selection (AMS)** | 60-80% | Smarter data selection -- pick examples where the Pareto frontier disagrees instead of random epoch-shuffled batches. Zero extra cost, just a better BatchSampler. |
| **#4 Contrastive Synthesis Reflection (CSR)** | 80-90% | Better reflection context -- one cheap ~500-token call distills contrastive pairs into actionable principles before the reflection LLM sees them. Extends CR, our strongest finding. |

### Standalone Non-Nature (2) -- replace GEPA entirely

| Method | Cost vs GEPA | Paradigm |
|--------|-------------|----------|
| **A3 Tournament Selection (PTS)** | 9% | Generate 64 diverse prompts, single-elimination bracket on shared 5-example subsets. 630 rollouts total. Comparative evaluation, no iteration. |
| **A5 Modular Decomposition (PDMO)** | 30-40% | Decompose prompt into 4 independent modules, optimize each on a small budget, compose + short joint refinement. Sum of small searches < one big search. |

### Standalone Nature-Inspired (4) -- biologically-inspired alternatives

| Method | Cost vs GEPA | Inspiration |
|--------|-------------|-------------|
| **B1 Slime Mold Pruning (SMNO)** | 8% | Physarum polycephalum. Generate 20 diverse prompts, evaluate all, flow resources to high-scorers, prune low-scorers. Progressive rounds: 20->10->5->3->1. ~540 rollouts. |
| **B5 Synaptic Pruning (SPDO)** | 3-7% | Brain development. Start with a maximally detailed ~2000-word prompt, ablate one section at a time, prune non-contributing parts. 30-50 evaluations. Inverse of GEPA's additive approach. |
| **B6 Ecological Succession (ESO)** | 50% | Forest succession. Curriculum learning: optimize on easy examples first (fast signal), then medium, then all. Each stage seeds the next. 3500 rollouts. |
| **B9 Ant Colony Components (ACPCO)** | 15-35% | Ant colony optimization. Decompose prompts into ~50 atomic components with pheromone levels. Sample components proportionally, evaluate, update pheromone. Uniquely solves credit assignment -- which sentence matters. |

### Summary

| # | Method | Type | Cost vs GEPA |
|---|--------|------|-------------|
| 1 | AMS | Stack on GEPA | 60-80% |
| 2 | CSR | Stack on GEPA | 80-90% |
| 3 | PTS | Standalone | 9% |
| 4 | PDMO | Standalone | 30-40% |
| 5 | SMNO | Nature standalone | 8% |
| 6 | SPDO | Nature standalone | 3-7% |
| 7 | ESO | Nature standalone | 50% |
| 8 | ACPCO | Nature standalone | 15-35% |
