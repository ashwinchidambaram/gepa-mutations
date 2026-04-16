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
import os
import re
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gepa_mutations.config import Settings
from gepa_mutations.base import build_reflection_lm
from gepa_mutations.benchmarks.loader import load_benchmark
from gepa_mutations.runner.experiment import BENCHMARK_SEED_PROMPTS


def get_task_description(benchmark: str) -> str:
    """Get task description for a benchmark."""
    return BENCHMARK_SEED_PROMPTS.get(benchmark, f"Solve {benchmark} task.")


def format_examples(examples: list, max_examples: int = 10) -> str:
    """Format benchmark examples for discovery prompt (truncate for readability)."""
    formatted = []
    for i, ex in enumerate(examples[:max_examples]):
        # Try to get a string representation of the example
        if hasattr(ex, "__dict__"):
            ex_dict = ex.__dict__
            # Extract key fields (input/question + expected output)
            ex_str = " | ".join(f"{k}: {str(v)[:100]}" for k, v in ex_dict.items())
        else:
            ex_str = str(ex)[:200]
        # Truncate each example to 200 chars for markdown readability
        ex_str = ex_str[:200]
        formatted.append(f"{i+1}. {ex_str}")
    return "\n".join(formatted)


def discover_skills_fixed_k(
    lm,
    benchmark: str,
    examples: list,
    task_description: str,
    k: int,
) -> tuple[list[str], str]:
    """Run fixed-K discovery on benchmark examples.

    Returns: (skill_names, raw_response)
    """
    # Format examples
    examples_text = format_examples(examples, max_examples=10)

    # Build fixed-K discovery prompt
    prompt = f"""Task: Identify the distinct skills or capabilities required to solve the following task.

Benchmark: {benchmark}
Task description: {task_description}

Examples from this benchmark:
{examples_text}

Identify exactly {k} distinct skills or reasoning capabilities needed to solve these examples well.
For each skill, provide:
  - Name (2-3 words)
  - Brief description (1 sentence)
  - Failure pattern (what going wrong looks like)

Format as a numbered list:
1. <skill name>: <description> Failure: <failure pattern>
2. ...
"""

    # Call LM
    try:
        response = lm(prompt)
    except Exception as e:
        print(f"  Warning: LM call failed: {e}", file=sys.stderr)
        return [], ""

    # Parse numbered list
    pattern = r'^\s*(\d+)[.)]\s+(.+?)(?=\n\s*\d+[.)]|\Z)'
    matches = list(re.finditer(pattern, response, re.MULTILINE | re.DOTALL))

    if not matches:
        print(f"  Warning: Could not parse skills from response", file=sys.stderr)
        return [], response

    # Extract skills
    skills = []
    for match in matches:
        skill_text = match.group(2).strip()
        # Extract just the skill name (before the colon)
        if ":" in skill_text:
            skill_name = skill_text.split(":")[0].strip()
        else:
            skill_name = skill_text.split("\n")[0].strip()
        skills.append(skill_name)

    return skills, response


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

                # Run discovery
                task_desc = get_task_description(benchmark)
                skills, raw_response = discover_skills_fixed_k(
                    lm, benchmark, examples, task_desc, args.k
                )
                print(f"{len(skills)} skills parsed")

                # Write to markdown
                f.write(f"## {benchmark} / seed {seed}\n\n")

                f.write("### Task description\n\n")
                f.write(f"{task_desc}\n\n")

                f.write("### Examples shown to discovery LLM\n\n")
                examples_text = format_examples(examples, max_examples=10)
                # Escape markdown special chars minimally
                f.write(f"{examples_text}\n\n")

                f.write("### Raw LLM output\n\n")
                f.write("```\n")
                f.write(raw_response[:2000])  # Limit to 2000 chars for readability
                if len(raw_response) > 2000:
                    f.write(f"\n... ({len(raw_response)} total chars)\n")
                f.write("```\n\n")

                f.write("### Parsed skills\n\n")
                if skills:
                    for i, skill in enumerate(skills, 1):
                        f.write(f"{i}. {skill}\n")
                else:
                    f.write("(no skills parsed — check raw output above)\n")

                f.write("\n---\n\n")

    print(f"\nWrote: {md_path}")
    print(f"\nReview {md_path} to assess discovery quality.")
    print(f"Look for: benchmark-specific, concrete skills (good) vs generic skills (bad)")


if __name__ == "__main__":
    main()
