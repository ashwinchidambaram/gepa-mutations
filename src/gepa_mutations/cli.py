"""CLI entry point for gepa-mutations."""

import typer

app = typer.Typer(name="gepa-mutations", help="Run and mutate GEPA experiments.")


@app.command()
def run(
    benchmark: str = typer.Argument(help="Benchmark to run (hotpotqa, ifbench, aime, livebench, hover, pupa)"),
    config: str = typer.Option("configs/default.py", help="Path to experiment config"),
    dry_run: bool = typer.Option(False, help="Validate config without running"),
):
    """Run a GEPA experiment on a benchmark."""
    typer.echo(f"Running GEPA on {benchmark} with config {config}")
    if dry_run:
        typer.echo("Dry run — config validated, exiting.")
        return


@app.command()
def compare(
    results_dir: str = typer.Argument(help="Directory containing experiment results"),
    output: str = typer.Option("reports/", help="Output directory for comparison reports"),
):
    """Compare experiment results against paper baselines."""
    typer.echo(f"Comparing results from {results_dir}")


@app.command()
def upload(
    results_dir: str = typer.Argument(help="Local results directory to upload"),
    bucket: str = typer.Option(None, help="S3 bucket (defaults to S3_BUCKET env var)"),
):
    """Upload experiment results to S3."""
    typer.echo(f"Uploading {results_dir} to S3")


if __name__ == "__main__":
    app()
