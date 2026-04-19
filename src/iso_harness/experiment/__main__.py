"""CLI entry point for the ISO experiment orchestrator.

Usage:
    python -m iso_harness.experiment --config configs/pilot.yaml [--dry-run] [--strict-git]
"""

from __future__ import annotations

import argparse
import logging
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m iso_harness.experiment",
        description="ISO experiment orchestrator",
    )
    parser.add_argument(
        "--config",
        required=True,
        metavar="PATH",
        help="Path to YAML experiment config file",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        default=False,
        help="Accepted for forward-compatibility with shell scripts (no-op; "
             "smoke test is configured in the YAML config itself)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Validate config and print run matrix without executing",
    )
    parser.add_argument(
        "--strict-git",
        action="store_true",
        default=False,
        help="Refuse to run if git working tree has uncommitted changes",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    from iso_harness.experiment.config import load_config
    from iso_harness.experiment.orchestrator import Orchestrator

    config = load_config(args.config)
    orchestrator = Orchestrator(config)
    matrix = orchestrator.build_matrix()

    if args.dry_run:
        orchestrator.dry_run(matrix)
        sys.exit(0)

    results = orchestrator.execute(matrix, run_fn=None, strict_git=args.strict_git)

    # Print summary
    total = len(results)
    completed = sum(1 for r in results if r.get("status") == "completed")
    skipped = sum(1 for r in results if r.get("status") == "skipped")
    failed = sum(1 for r in results if r.get("status") == "failed")
    budget_exhausted = sum(1 for r in results if r.get("status") == "budget_exhausted")

    print()
    print(f"=== Orchestrator Summary ===")
    print(f"  Total runs:       {total}")
    print(f"  Completed:        {completed}")
    print(f"  Skipped:          {skipped}")
    print(f"  Budget exhausted: {budget_exhausted}")
    print(f"  Failed:           {failed}")


if __name__ == "__main__":
    main()
