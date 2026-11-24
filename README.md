# laropt

Tune **any FHiCL configuration** of a LArSoft job against a metric you define — from a YAML file on the
command line, or from a notebook.

```
optimizer.ask() ─► write FHiCLs ─► executor (local / grid) ─► lar ─► metric ─► objective ─► optimizer.tell()
                                      └── `laropt run-point` per point ──┘        results → writer (sqlite/csv/…)
```

## Install

```bash
pip install -e .[all]        # numpy pandas scipy pyyaml click (+ matplotlib, scikit-learn, uproot, pytest)
```
ROOT/LArSoft are only needed on the machine where the metric runs (`root_macro`, `lar`).

## Quick start

```bash
laropt init -o study.yaml            # annotated example (port of the old hit-tuning script)
laropt list                          # every registered optimizer / executor / metric / objective / writer / plot
laropt validate -c study.yaml [--fhicl-dump]
laropt run -c study.yaml             # full loop; safe to re-run: it resumes
laropt status|best|plot -c study.yaml
laropt run -c study.yaml -s optimizer.name=pso -s optimizer.n_particles=16   # override any key
```

Notebook:

```python
from laropt import Study
study = Study("study.yaml", overrides={"optimizer": {"name": "bayes", "n_initial": 20}})
study.run()
study.points().sort_values("objective").head()     # one row per evaluation (params, objective, summaries)
study.events()                                     # per-event metric table;  study.merged() joins both
study.plot("convergence"); study.plot("slices", columns=["chi2NDF"]); study.plot("scatter2d", x="a", y="b")
study.reduce({"filters": ["plane2NOfHits > 0"], "aggregation": {"mean": ["eventRatio"]}})
Study.open("out/hit_tuning").best(3)               # reopen a finished/ongoing study
```

## Everything is registered

| kind | built in | add your own |
|---|---|---|
| `optimizer` | `lhs sobol halton random grid pso bayes` | subclass `laropt.optimizers.Optimizer` (`ask`/`tell`/`done`, unit cube, minimising) |
| `executor` | `local`, `grid` | subclass `laropt.executors.Executor` (`launch`, `wait`) |
| `metric` | `root_macro`, `fcl_analyzer`, `python` | subclass `laropt.metrics.Metric` (`evaluate(ctx) -> DataFrame`) |
| `objective` | `expression`, `callable` | subclass `laropt.objectives.Objective` |
| `writer` | `sqlite csv pickle parquet` | subclass `laropt.writers.ResultWriter` |
| `plot` | `convergence slices parallel correlation scatter2d events_hist` | subclass `laropt.plotting.Plot` |

```python
from laropt import register
from laropt.optimizers import Optimizer

@register("optimizer", "my-cma")
class MyCMA(Optimizer): ...
```
In YAML put the module or `.py` file under `plugins:` (or publish a `laropt.optimizers` entry point) and use
`optimizer: {name: my-cma}`. Anything defined in a notebook works from the notebook; for subprocess/grid workers
it must be importable, i.e. in a plugin file (or use `executor: {in_process: true}`).

## Tuning any FHiCL

`parameters:` maps search dimensions to FHiCL paths; laropt appends `path: value` overrides to a copy of your template
(it never edits it), so anything reachable by a dotted path works, including values pulled in through `#include`.

```yaml
parameters:
  roiThreshold:
    bounds: [3, 7]
    size: 3                       # vector; independent per element unless `tied: true`
    per_element: true             # element i -> path containing {i}
    expand: {producer: [gaushit2dTPCEE, gaushit2dTPCWW]}      # one write per value
    paths: ["physics.producers.{producer}.HitFinderToolVec.CandidateHitsPlane{i}.RoiThreshold"]
  chi2NDF: {bounds: [200, 2500], log: true, paths: ["physics.producers.hits.Chi2NDF"]}
  maxMultiHit: {type: int, bounds: [5, 10], paths: [...]}      # uniform over integers
  algo: {choices: [A, B], paths: [...]}                        # categorical
```
Constants for every job go under `overrides:` (strings are quoted; use `raw:` to write them verbatim).

> Behaviour change vs. the old script: vectors used to get *one* value copied to all planes. Now each element is its own
> dimension; add `tied: true` to reproduce the old behaviour.

## Metrics and objectives

The **metric** turns a tuned `lar` output into a table (one row per event); the **objective** reduces that table to one number
per grid point.

* `root_macro` — C macro compiled with ACLiC, called `function(lar_output, hist_file)`, returns `vector<vector<double>>`;
  name the columns in `columns:`. Each point compiles a private copy (`isolate: true`) so parallel jobs don't race.
* `fcl_analyzer` — a second FHiCL (analyzer/ntuple maker) run on the tuned output; its TFileService TTree (via uproot or
  ROOT) or a CSV/JSON/parquet file is read back. Parameters can be injected with `overrides:`.
* `python` — `function: "my_module:score"`, `score(ctx) -> DataFrame`.
* `objective.expr` sees `df`, `np`, `pd`, `params` and every column; `summaries:` stores extra columns; `direction: maximize` is supported.

## Running on a grid (and adaptive optimizers)

Every point is evaluated by `laropt run-point -c <workdir>/config.yaml -p <id>`, which only writes files into
`<workdir>/runs/point_xxxxxx/` — workers never touch the database, so any scheduler works. The `grid` executor writes
`jobs/genNNN/commands.txt` + `job.sh` (array-task id selects the line) and either

* submits and polls (`submit: "sbatch --array=0-{n_last} {script}"`, `condor_submit`, …), or
* with no `submit:`, just prepares the files: submit them yourself, then `laropt collect` / `laropt run` again.

Optimizers work in generations (`ask` → run all → `tell`), so adaptive ones (PSO, Bayesian) use the grid the same way
LHS does: one array job per generation. State is stored in the results table; on resume the optimizer is replayed
deterministically from the stored points (a paused grid study gives the same points as an uninterrupted one — covered by
a test). Failed points get a penalty loss (`failure.penalty`, default: worse than the worst seen) and are kept in the table.

## Layout of a study directory

```
<workdir>/config.yaml        resolved config (used by workers)
<workdir>/fcls/genNNN/*.fcl  generated FHiCLs
<workdir>/runs/point_*/      point.json, lar logs, events.csv, result.json
<workdir>/jobs/genNNN/       grid scripts
<workdir>/results.db         `points` and `events` tables (or results.points.csv, …)
```

Legacy databases from the old script (`runs` table, `jobNum`) can be merged with `laropt merge -i dir -o out.pkl -r reduce.yaml`
(same `filters / pre_transforms / aggregation / post_transforms` YAML).

## Not covered / caveats

* `root_macro`, `fcl_analyzer` ROOT reading and `--fhicl-dump` are written against the real tools but **could not be run
  here** (no ROOT/LArSoft); the test-suite exercises the whole pipeline with a fake `lar`, a Python metric and a CSV analyzer.
* `fhicl-dump` output parsing is best effort and only produces warnings.
* The metric's per-point work is serial inside one worker; parallelism is across points.
