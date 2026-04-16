#!/usr/bin/env python3
"""preflight_discovery_quality.py

Run discovery 3 times on 3 benchmarks and dump outputs for MANUAL REVIEW.
Helps assess if LLM produces benchmark-specific, concrete skills (good)
or generic/uniform skills (bad — means we need a bigger reflection model).

Usage:
    python scripts/preflight_discovery_quality.py \\
        --benchmarks hotpotqa hover pupa \\
        --seeds 42 123 456 \\
        --output-dir experiments/03-inductive-discovery-2026-04/logs \\
        --k 5
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
# Add slime_mold to path (so we can import the production discovery function)
sys.path.insert(0, str(Path(__file__).parent.parent / "methods" / "slime_mold"))

from gepa_mutations.config import Settings
from gepa_mutations.base import build_reflection_lm
from gepa_mutations.benchmarks.loader import load_benchmark
from gepa_mutations.runner.experiment import (
    BENCHMARK_OUTPUT_SHAPES,
    BENCHMARK_TASK_INSTRUCTIONS,
)
from slime_mold.colony import discover_strategies


def format_examples(examples: list, max_examples: int = 10) -> str:
    """Format benchmark examples for display in the markdown report.

    Mirrors the extraction used by production discover_strategies() in colony.py:
    tries .input, .question, then falls back to str(ex).
    """
    formatted = []
    for i, ex in enumerate(examples[:max_examples]):
        input_str = getattr(ex, "input", None) or getattr(ex, "question", None) or str(ex)
        answer_str = getattr(ex, "answer", None) or getattr(ex, "output", "") or ""
        formatted.append(
            f"Example {i+1}:\n"
            f"  Input: {str(input_str)[:500]}\n"
            f"  Expected answer: {str(answer_str)[:200]}"
        )
    return "\n\n".join(formatted)


def discover_skills_fixed_k(
    lm,
    benchmark: str,
    examples: list,
    k: int,
) -> tuple[list[str], list[str], str, bool]:
    """Run fixed-K discovery on benchmark examples via the production function.

    Returns: (skill_names, technique_names, raw_response, fallback_used)
    """
    task_description = BENCHMARK_TASK_INSTRUCTIONS.get(
        benchmark, "Solve the examples correctly."
    )
    output_shape = BENCHMARK_OUTPUT_SHAPES.get(
        benchmark, "a response in the format shown by the examples"
    )

    try:
        strategies, raw_response, fallback = discover_strategies(
            reflection_lm=lm,
            benchmark=benchmark,
            task_description=task_description,
            output_shape=output_shape,
            examples=examples,
            k=k,
        )
    except Exception as e:
        print(f"  Warning: discover_strategies failed: {e}", file=sys.stderr)
        return [], [], "", False

    skill_names = [s.name for s in strategies]
    techniques = [s.technique for s in strategies]
    return skill_names, techniques, raw_response, fallback


def main():
    parser = argparse.ArgumentParser(
        description="Run discovery on benchmarks and dump outputs for manual review"
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["hotpotqa", "hover", "pupa"],
        help="Benchmarks to probe (default: hotpotqa hover pupa)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 123, 456],
        help="Random seeds for probes (default: 42 123 456)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/03-inductive-discovery-2026-04/logs",
        help="Output directory for results",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Fixed K value for discovery (default: 5)",
    )
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load LM
    settings = Settings()
    lm = build_reflection_lm(settings)

    print(f"=== Discovery Quality Preflight ===")
    print(f"Model: {settings.gepa_model}")
    print(f"K: {args.k}")
    print(f"Benchmarks: {', '.join(args.benchmarks)}")
    print(f"Seeds: {', '.join(map(str, args.seeds))}")
    print(f"Output dir: {output_dir}")
    print()

    # Markdown output
    md_path = output_dir / "discovery_preflight.md"
    with open(md_path, "w") as f:
        f.write("# Discovery Quality Preflight\n\n")
        f.write(f"**Generated:** {datetime.now().isoformat()}\n\n")
        f.write(f"**Model:** {settings.gepa_model}\n\n")
        f.write(f"**K:** {args.k}\n\n")
        f.write("---\n\n")

        # Run probes
        for benchmark in args.benchmarks:
            print(f"Probing {benchmark}...")

            for seed in args.seeds:
                print(f"  seed {seed}...", end=" ", flush=True)

                # Load examples
                try:
                    data = load_benchmark(benchmark, seed=seed)
                    examples = data.train[:10]  # Use first 10 train examples
                except Exception as e:
                    print(f"Error: {e}")
                    continue

                # Run discovery (calls the production discover_strategies function)
                skills, techniques, raw_response, fallback = discover_skills_fixed_k(
                    lm, benchmark, examples, args.k
                )
                fallback_tag = " (FELL BACK TO PRESCRIBED)" if fallback else ""
                print(f"{len(skills)} skills parsed{fallback_tag}")

                # Write to markdown
                f.write(f"## {benchmark} / seed {seed}\n\n")

                f.write("### Discovery context\n\n")
                f.write(f"- **Task instruction:** {BENCHMARK_TASK_INSTRUCTIONS.get(benchmark, '(default)')}\n")
                f.write(f"- **Output shape:** {BENCHMARK_OUTPUT_SHAPES.get(benchmark, '(default)')}\n")
                if fallback:
                    f.write(f"- **Fallback used:** YES — discovery produced too few skills, fell back to PRESCRIBED_STRATEGIES\n")
                f.write("\n")

                f.write("### Examples shown to discovery LLM\n\n")
                examples_text = format_examples(examples, max_examples=10)
                f.write(f"{examples_text}\n\n")

                f.write("### Raw LLM output\n\n")
                f.write("```\n")
                f.write(raw_response[:3000])  # Limit for readability
                if len(raw_response) > 3000:
                    f.write(f"\n... ({len(raw_response)} total chars)\n")
                f.write("```\n\n")

                f.write("### Parsed skills (with techniques)\n\n")
                if skills:
                    for i, (skill, tech) in enumerate(zip(skills, techniques), 1):
                        tech_label = f" — *{tech}*" if tech else ""
                        f.write(f"{i}. {skill}{tech_label}\n")
                else:
                    f.write("(no skills parsed — check raw output above)\n")

                f.write("\n---\n\n")

    print(f"\nWrote: {md_path}")
    print(f"\nReview {md_path} to assess discovery quality.")
    print(f"Look for: benchmark-specific, concrete skills (good) vs generic skills (bad)")


if __name__ == "__main__":
    main()
