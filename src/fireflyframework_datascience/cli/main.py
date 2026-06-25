# Copyright 2026 Firefly Software Foundation.
"""``firefly-ds`` command-line entry point."""

from __future__ import annotations

import importlib.util
import platform

import click
from rich.console import Console
from rich.table import Table

from fireflyframework_datascience import FireflyDataScienceApplication, __version__

console = Console()

# Optional-extra → representative importable modules. Used by ``doctor``.
_EXTRAS: dict[str, list[str]] = {
    "tabular": ["pandas", "numpy", "sklearn", "xgboost", "lightgbm", "catboost", "optuna"],
    "tabfm": ["tabpfn"],
    "automl": ["autogluon"],
    "dl": ["torch", "lightning", "torchvision", "accelerate"],
    "nlp": ["transformers", "datasets", "peft", "trl"],
    "tracking": ["mlflow"],
    "validation": ["pandera"],
    "featurestore": ["feast"],
    "serving": ["bentoml"],
    "lineage": ["openlineage"],
    "orchestration": ["airflow"],
    "data": ["openml", "polars"],
    "genai": ["fireflyframework_agentic", "pydantic_monty"],
}


def _available(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        return False


@click.group()
@click.version_option(__version__, prog_name="firefly-ds")
def cli() -> None:
    """Firefly DataScience — AutoML that fuses GenAI with classical ML & Deep Learning."""


@cli.command()
def version() -> None:
    """Print the framework version."""
    console.print(f"Firefly DataScience [bold cyan]v{__version__}[/bold cyan]")


@cli.command()
def doctor() -> None:
    """Check the environment and report which adapter extras are installed."""
    console.print(f"[bold cyan]Firefly DataScience[/bold cyan] doctor — v{__version__}")
    console.print(f"  python : {platform.python_version()} ({platform.platform()})")
    agentic_ok = _available("fireflyframework_agentic")
    console.print(f"  agentic: {'[green]ok[/green]' if agentic_ok else '[red]MISSING[/red]'} (required)")

    table = Table(title="Optional adapter extras", show_lines=False)
    table.add_column("extra", style="cyan")
    table.add_column("status")
    table.add_column("modules")
    for extra, modules in _EXTRAS.items():
        present = [m for m in modules if _available(m)]
        if not present:
            status = "[dim]not installed[/dim]"
        elif len(present) == len(modules):
            status = "[green]installed[/green]"
        else:
            status = "[yellow]partial[/yellow]"
        table.add_row(extra, status, f"{len(present)}/{len(modules)}")
    console.print(table)


@cli.command()
@click.option("--config-dir", default=None, help="Directory containing firefly-datascience.yaml.")
@click.option("--profile", "profiles", multiple=True, help="Active profile (repeatable).")
def introspect(config_dir: str | None, profiles: tuple[str, ...]) -> None:
    """Boot the application and show discovered auto-configurations and registered beans."""
    ctx = FireflyDataScienceApplication.run(
        config_dir=config_dir,
        profiles=list(profiles) or None,
        print_output=False,
    )
    console.print(f"[bold cyan]Firefly DataScience[/bold cyan] introspection — v{__version__}")
    console.print(f"  profiles    : {', '.join(ctx.config.profiles) or 'default'}")
    console.print(f"  beans       : {ctx.bean_count}")
    console.print(f"  ml framework: {ctx.config.default_ml_framework}")
    console.print(f"  genai       : {'enabled' if ctx.config.genai.enabled else 'disabled'}")

    table = Table(title="Applied auto-configurations")
    table.add_column("class", style="cyan")
    table.add_column("module")
    for ac in ctx.applied_auto_configurations:
        table.add_row(ac.__name__, ac.__module__)
    console.print(table)

    console.print(f"  registered beans: {', '.join(ctx.container.bean_names())}")


if __name__ == "__main__":
    cli()
