"""Command line interface: everything is driven by a YAML study file."""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click
import pandas as pd
import yaml

from . import registry as reg
from .reduce import reduce_frame
from .study import Study


def _overrides(sets):
    out = {}
    for s in sets:
        if "=" not in s:
            raise click.BadParameter(f"--set expects key=value, got '{s}'")
        k, v = s.split("=", 1)
        out[k.strip()] = yaml.safe_load(v)
    return out


def config_options(f):
    f = click.option("-s", "--set", "sets", multiple=True, metavar="KEY=VALUE",
                     help="Override a config entry, e.g. -s optimizer.n_samples=50")(f)
    f = click.option("-c", "--config", required=True, type=click.Path(),
                     help="Study YAML file (or an existing study work directory)")(f)
    return f


def _open(config, sets) -> Study:
    return Study.open(config, overrides=_overrides(sets))


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Debug logging")
def cli(verbose):
    """laropt - optimise any FHiCL configuration against a metric."""
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, stream=sys.stderr,
                        format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")


@cli.command("list")
def list_components():
    """List registered optimizers, executors, metrics, objectives, writers and plots."""
    for kind, r in reg._REGISTRIES.items():
        click.echo(f"{kind}s:")
        for name in r.names():
            doc = (r.get(name).__doc__ or "").strip().splitlines()
            click.echo(f"  {name:14s} {doc[0] if doc else ''}")


@cli.command("init")
@click.option("-o", "--output", default="study.yaml", type=click.Path(), show_default=True)
def init(output):
    """Write an example study YAML to start from."""
    if Path(output).exists():
        raise click.ClickException(f"{output} already exists")
    Path(output).write_text((Path(__file__).parent / "data" / "example.yaml").read_text())
    click.echo(f"wrote {output}")


@cli.command()
@config_options
@click.option("--fhicl-dump", "use_dump", is_flag=True, help="Check parameter paths against `fhicl-dump` of the template")
def validate(config, sets, use_dump):
    """Build every component and sanity-check the configuration."""
    study = _open(config, sets)
    click.echo(study.space)
    warnings = study.validate(use_dump)
    for w in warnings:
        click.echo(f"WARNING: {w}")
    click.echo("config OK" if not warnings else f"{len(warnings)} warning(s)")


@cli.command()
@config_options
def create(config, sets):
    """Generate the next generation of FHiCL files without running anything."""
    study = _open(config, sets)
    pts = study.create()
    click.echo(f"{len(pts)} point(s) ready in {study.workdir / 'fcls'}")


@cli.command()
@config_options
@click.option("-g", "--max-generations", type=int, default=None, help="Stop after this many new generations")
def run(config, sets, max_generations):
    """Run (or resume) the optimisation loop."""
    study = _open(config, sets)
    click.echo(json.dumps(study.run(max_generations), indent=2))


@cli.command("run-point")
@config_options
@click.option("-p", "--point", "point_id", required=True, type=int, help="Point id to evaluate")
@click.option("--strict", is_flag=True, help="Exit non-zero if the point failed")
def run_point_cmd(config, sets, point_id, strict):
    """Worker entry point: evaluate ONE point (what grid jobs execute)."""
    from .worker import run_point
    res = run_point(_open(config, sets), point_id)
    click.echo(f"point {point_id}: {res['status']} ({res['duration']:.1f}s)")
    if res["status"] != "done" and strict:
        sys.exit(1)


@cli.command()
@config_options
def collect(config, sets):
    """Ingest finished point results into the results tables."""
    click.echo(json.dumps(_open(config, sets).collect(), indent=2))


@cli.command()
@config_options
def status(config, sets):
    """Show study progress."""
    click.echo(json.dumps(_open(config, sets).status(), indent=2))


@cli.command()
@config_options
@click.option("-n", default=5, show_default=True, help="Number of points to show")
def best(config, sets, n):
    """Print the best evaluated points."""
    study = _open(config, sets)
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        click.echo(study.best(n).drop(columns=["unit_x", "fcl", "run_dir", "error"], errors="ignore").to_string(index=False))


@cli.command()
@config_options
@click.option("-n", "--name", default=None, help="Plot to draw (default: those listed under 'plots:')")
@click.option("-o", "--output", default=None, help="Output file (relative to <workdir>/plots)")
def plot(config, sets, name, output):
    """Draw plots to files."""
    import matplotlib
    matplotlib.use("Agg")
    study = _open(config, sets)
    if name:
        study.plot(name, save=output or f"{name}.png")
        click.echo(f"wrote {study.workdir / 'plots' / (output or name + '.png')}")
    else:
        for p in study.make_plots():
            click.echo(f"wrote {p}")


@cli.command()
@click.option("-i", "--input", "inputs", multiple=True, required=True, type=click.Path(exists=True),
              help="Study dirs, directories of .db files, or .db files (legacy 'runs' tables ok)")
@click.option("-o", "--output", required=True, help="Output .pkl / .csv / .db")
@click.option("-r", "--reduce", "reduce_cfg", default=None, type=click.Path(exists=True),
              help="YAML with filters / pre_transforms / aggregation / post_transforms")
@click.option("--no-aggregate", is_flag=True)
def merge(inputs, output, reduce_cfg, no_aggregate):
    """Merge several studies / databases into one table, optionally reduced."""
    import sqlite3
    frames = []
    for inp in inputs:
        p = Path(inp)
        if p.is_dir() and (p / "config.yaml").exists():
            df = Study.open(p).merged()
            df.insert(0, "study", p.name)
            frames.append(df)
            continue
        for f in ([p] if p.is_file() else sorted(p.rglob("*.db"))):
            with sqlite3.connect(f) as conn:
                try:
                    df = pd.read_sql("SELECT * FROM runs", conn).rename(columns={"jobNum": "point_id"})
                except Exception:
                    try:
                        pts = pd.read_sql("SELECT * FROM points", conn)
                        ev = pd.read_sql("SELECT * FROM events", conn)
                        df = ev.merge(pts.drop(columns=["status", "fcl", "run_dir", "error", "unit_x"], errors="ignore"),
                                      on="point_id")
                    except Exception as exc:
                        click.echo(f"skipping {f}: {exc}")
                        continue
            df.insert(0, "study", p.name if p.is_dir() else f.stem)
            frames.append(df)
    if not frames:
        raise click.ClickException("nothing to merge")
    full = pd.concat(frames, ignore_index=True)
    click.echo(f"loaded {len(full)} rows")
    if reduce_cfg:
        conf = yaml.safe_load(Path(reduce_cfg).read_text())
        full = reduce_frame(full, conf, key=["study", "point_id"], aggregate=not no_aggregate)
    if output.endswith(".pkl"):
        full.to_pickle(output)
    elif output.endswith(".csv"):
        full.to_csv(output, index=False)
    else:
        with sqlite3.connect(output) as conn:
            full.to_sql("runs", conn, if_exists="replace", index=False)
    click.echo(f"wrote {len(full)} rows to {output}")


def main():
    cli()


if __name__ == "__main__":
    main()
