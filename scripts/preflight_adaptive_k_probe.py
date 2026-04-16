#!/usr/bin/env python3
"""preflight_adaptive_k_probe.py

Probe adaptive-K discovery (k=None, LLM decides) across benchmarks.
Runs 3 seeds on each of 4 benchmarks, outputs K values + markdown table.

Usage:
    python scripts/preflight_adaptive_k_probe.py \\
        --benchmarks hotpotqa hover pupa ifbench \\
        --seeds 42 123 456 \\
        --output-dir experiments/03-inductive-discovery-2026-04/logs
"""

import argparse
import json
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
    """Format benchmark examples for discovery prompt."""
    formatted = []
    for i, ex in enumerate(examples[:max_examples]):
        # Try to get a string representation of the example
        if hasattr(ex, "__dict__"):
            ex_dict = ex.__dict__
            # Extract key fields (input/question + expected output)
            ex_str = " | ".join(f"{k}: {str(v)[:100]}" for k, v in ex_dict.items())
        else:
            ex_str = str(ex)[:200]
        formatted.append(f"{i+1}. {ex_str}")
    return "\n".join(formatted)


def discover_skills_adaptive(
    lm,
    benchmark: str,
    examples: list,
    task_description: str,
) -> tuple[int, list[str], str]:
    """Run adaptive-K discovery on benchmark examples.

    Returns: (k_value, skill_names, raw_response)
    """
    # Format examples
    examples_text = format_examples(examples, max_examples=10)

    # Build adaptive-K discovery prompt (LLM decides)
    prompt = f"""Task: Identify the distinct skills or capabilities required to solve the following task.

Benchmark: {benchmark}
Task description: {task_description}

Examples from this benchmark:
{examples_text}

Identify the number of skills you actually see (typically 3-10) distinct skills or reasoning capabilities needed to solve these examples well.
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
        return None, [], ""

    # Parse numbered list
    # Pattern: line starts with digit(s), followed by . or ), then skill text
    pattern = r'^\s*(\d+)[.)]\s+(.+?)(?=\n\s*\d+[.)]|\Z)'
    matches = list(re.finditer(pattern, response, re.MULTILINE | re.DOTALL))

    if not matches:
        print(f"  Warning: Could not parse skills from response", file=sys.stderr)
        return None, [], response

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

    k = len(skills)
    return k, skills, response


def main():
    parser = argparse.ArgumentParser(
        description="Probe adaptive-K discovery across benchmarks"
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["hotpotqa", "hover", "pupa", "ifbench"],
        help="Benchmarks to probe (default: hotpotqa hover pupa ifbench)",
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
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load LM
    settings = Settings()
    lm = build_reflection_lm(settings)

    print(f"=== Adaptive K Discovery Probe ===")
    print(f"Model: {settings.gepa_model}")
    print(f"Benchmarks: {', '.join(args.benchmarks)}")
    print(f"Seeds: {', '.join(map(str, args.seeds))}")
    print(f"Output dir: {output_dir}")
    print()

    # Run probes
    probes = []
    k_per_benchmark = {}

    for benchmark in args.benchmarks:
        print(f"Probing {benchmark}...")
        k_per_benchmark[benchmark] = []

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
            k, skills, raw_response = discover_skills_adaptive(
                lm, benchmark, examples, task_desc
            )

            if k is not None:
                print(f"k={k}")
                k_per_benchmark[benchmark].append(k)
                probes.append({
                    "benchmark": benchmark,
                    "seed": seed,
                    "k": k,
                    "skills": skills,
                })
            else:
                print("FAILED (parse error)")

    # Compute summary statistics
    summary = {
        "per_benchmark_k_range": {},
        "global_k_range": {},
        "decision_hint": "",
    }

    all_k_values = []
    for benchmark, k_values in k_per_benchmark.items():
        if k_values:
            summary["per_benchmark_k_range"][benchmark] = {
                "min": min(k_values),
                "max": max(k_values),
                "mean": sum(k_values) / len(k_values),
            }
            all_k_values.extend(k_values)

    if all_k_values:
        global_min = min(all_k_values)
        global_max = max(all_k_values)
        spread = global_max - global_min

        summary["global_k_range"] = {
            "min": global_min,
            "max": global_max,
            "spread": spread,
        }

        if spread >= 2:
            summary["decision_hint"] = (
                f"K varies (range {spread}) — adaptive K probably adapts; "
                f"consider running adaptive K arm in Phase 10"
            )
        else:
            summary["decision_hint"] = (
                f"K does not vary much (range {spread}) — "
                f"K~{global_min}-{global_max} is stable; skip K=adaptive arm"
            )

    # Write JSON
    results = {
        "timestamp": datetime.now().isoformat(),
        "model": settings.gepa_model,
        "probes": probes,
        "summary": summary,
    }

    json_path = output_dir / "adaptive_k_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote: {json_path}")

    # Write markdown table
    md_path = output_dir / "adaptive_k_results.md"
    with open(md_path, "w") as f:
        f.write("# Adaptive K Discovery Probe\n\n")
        f.write(f"**Generated:** {datetime.now().isoformat()}\n\n")
        f.write(f"**Model:** {settings.gepa_model}\n\n")

        # Results table
        f.write("## Results\n\n")
        f.write("| Benchmark | Seed | K | Skills |\n")
        f.write("|-----------|------|---|--------|\n")
        for probe in probes:
            skills_str = ", ".join(probe["skills"][:3])
            if len(probe["skills"]) > 3:
                skills_str += f", ... ({len(probe['skills'])} total)"
            f.write(
                f"| {probe['benchmark']} | {probe['seed']} | "
                f"{probe['k']} | {skills_str} |\n"
            )

        # Summary
        f.write("\n## Summary\n\n")
        if summary["per_benchmark_k_range"]:
            f.write("### Per-Benchmark K Range\n\n")
            for benchmark, stats in summary["per_benchmark_k_range"].items():
                f.write(
                    f"- **{benchmark}:** min={stats['min']}, "
                    f"max={stats['max']}, mean={stats['mean']:.1f}\n"
                )

        if summary["global_k_range"]:
            f.write("\n### Global K Range\n\n")
            stats = summary["global_k_range"]
            f.write(f"- **min:** {stats['min']}\n")
            f.write(f"- **max:** {stats['max']}\n")
            f.write(f"- **spread:** {stats['spread']}\n")

        f.write("\n### Decision Hint\n\n")
        f.write(f"{summary['decision_hint']}\n")

    print(f"Wrote: {md_path}")
    print(f"\nSummary: {summary['decision_hint']}")


if __name__ == "__main__":
    main()
