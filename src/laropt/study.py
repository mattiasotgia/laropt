"""The Study: orchestrates ask -> write FHiCLs -> execute -> collect -> tell.

State lives entirely in ``<workdir>`` (the ``points`` table written by the result
writer, the per-point run directories and the resolved ``config.yaml``), so a study
can be paused (e.g. while grid jobs run) and resumed from another process: on
resume the optimizer is replayed from the stored history.
"""
from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from functools import cached_property
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from .fcl import FclTemplate, check_assignments, fhicl_dump, flatten_dump, FclError
from .params import ParamSpace
from .reduce import reduce_frame
from .registry import EXECUTORS, METRICS, OBJECTIVES, OPTIMIZERS, PLOTS, WRITERS, load_plugins
from .results import ResultsView

log = logging.getLogger("laropt")

COMPONENT_KEYS = ("optimizer", "executor", "objective", "writer")
DEFAULTS: Dict[str, Any] = {
    "study": {"name": "laropt_study", "workdir": "laropt_out", "seed": 42},
    "overrides": {},
    "input": {"files": [], "batch_size": None},
    "lar": {"executable": "lar", "n_events": -1, "extra_args": [], "output": True, "keep_output": False,
            "tfile": None, "timeout": None},
    "optimizer": {"name": "lhs", "n_samples": 200},
    "executor": {"name": "local", "workers": 1},
    "objective": {"name": "expression", "direction": "minimize"},
    "writer": {"name": "sqlite"},
    "summaries": {},
    "failure": {"penalty": None},
    "plugins": [],
    "plots": [],
}
KNOWN_KEYS = set(DEFAULTS) | {"template", "parameters", "metric"}


def _deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def _set_dotted(d: dict, key: str, value):
    parts = key.split(".")
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = value


def load_config(config, base_dir=None, overrides: Optional[dict] = None):
    if isinstance(config, (str, Path)):
        path = Path(config).expanduser().resolve()
        raw = yaml.safe_load(path.read_text()) or {}
        file_dir = path.parent
    else:
        raw, file_dir = copy.deepcopy(dict(config)), Path.cwd()
    for k, v in (overrides or {}).items():
        _set_dotted(raw, k, v)
    saved = (raw.get("study") or {}).get("base_dir")
    base = Path(base_dir).resolve() if base_dir else Path(saved) if saved else file_dir

    unknown = set(raw) - KNOWN_KEYS
    if unknown:
        log.warning("ignoring unknown top-level config keys: %s", sorted(unknown))

    if isinstance(raw.get("input"), (str, list)):
        raw["input"] = {"files": raw["input"]}
    cfg: Dict[str, Any] = {}
    for key, default in DEFAULTS.items():
        if key in COMPONENT_KEYS:
            v = raw.get(key, default)
            v = {"name": v} if isinstance(v, str) else dict(v)
            v.setdefault("name", default["name"])
            cfg[key] = v
        elif isinstance(default, dict):
            cfg[key] = _deep_merge(default, raw.get(key) or {})
        else:
            cfg[key] = raw.get(key, default)
    m = raw.get("metric")
    cfg["metric"] = {"name": m} if isinstance(m, str) else m
    cfg["template"], cfg["parameters"] = raw.get("template"), raw.get("parameters")
    for req in ("template", "parameters", "metric"):
        if not cfg[req]:
            raise ValueError(f"config is missing required section '{req}'")
    if not isinstance(cfg["plots"], list):
        raise ValueError("'plots' must be a list")
    return cfg, base


@dataclass
class Point:
    id: int
    generation: int
    x: np.ndarray
    params: Dict[str, Any]
    fcl: Path
    run_dir: Path


class Study:
    """A tuning study. Build from a YAML path or a config dict::

        study = Study("study.yaml", overrides={"optimizer.n_samples": 20})
        study.run()
        study.points()            # DataFrame, one row per evaluation
        study.plot("convergence")
    """

    def __init__(self, config, *, base_dir=None, overrides: Optional[dict] = None):
        self.config, self.base_dir = load_config(config, base_dir, overrides)
        load_plugins(self.config["plugins"], self.base_dir)
        s = self.config["study"]
        self.name = s["name"]
        self.workdir = self._resolve(s["workdir"])
        self.space = ParamSpace.from_config(self.config["parameters"])
        self._synced = False

    @classmethod
    def open(cls, path, **kw) -> "Study":
        """Open a study from its YAML file or from an existing work directory."""
        p = Path(path)
        return cls(p / "config.yaml" if p.is_dir() else p, **kw)

    # ---- paths -------------------------------------------------------------------
    def _resolve(self, p) -> Path:
        p = Path(str(p)).expanduser()
        return p if p.is_absolute() else (self.base_dir / p).resolve()

    @property
    def config_path(self) -> Path:
        return self.workdir / "config.yaml"

    def run_dir(self, pid: int) -> Path:
        return self.workdir / "runs" / f"point_{pid:06d}"

    # ---- components (built lazily; the worker never builds the optimizer) -------------
    @cached_property
    def template(self) -> FclTemplate:
        return FclTemplate(self._resolve(self.config["template"]))

    @cached_property
    def optimizer(self):
        cfg = dict(self.config["optimizer"])
        name = cfg.pop("name")
        cfg.setdefault("seed", self.config["study"].get("seed"))
        return OPTIMIZERS.create(name, dims=self.space.dims, **cfg)

    @cached_property
    def executor(self):
        cfg = dict(self.config["executor"])
        return EXECUTORS.create(cfg.pop("name"), base_dir=self.base_dir, **cfg)

    @cached_property
    def metric(self):
        cfg = dict(self.config["metric"])
        return METRICS.create(cfg.pop("name"), base_dir=self.base_dir, **cfg)

    @cached_property
    def objective(self):
        cfg = dict(self.config["objective"])
        return OBJECTIVES.create(cfg.pop("name"), base_dir=self.base_dir, summaries=self.config["summaries"], **cfg)

    @cached_property
    def writer(self):
        cfg = dict(self.config["writer"])
        return WRITERS.create(cfg.pop("name"), path=self.workdir / "results", **cfg)

    @property
    def direction(self) -> str:
        return self.objective.direction

    # ---- inputs ----------------------------------------------------------------------
    def input_files(self) -> List[str]:
        spec = self.config["input"]["files"]
        items = [spec] if isinstance(spec, (str, Path)) else list(spec or [])
        files: List[str] = []
        for it in map(str, items):
            if "://" in it:
                files.append(it)
            elif any(ch in it for ch in "*?["):
                files += sorted(glob(str(self._resolve(it))))
            elif it.endswith(".root"):
                files.append(str(self._resolve(it)))
            else:  # a list file, one path per line
                lines = [l.strip() for l in self._resolve(it).read_text().splitlines()]
                files += [l for l in lines if l and not l.startswith("#")]
        return files

    def input_batches(self) -> List[List[str]]:
        files = self.input_files()
        size = self.config["input"].get("batch_size")
        if not files:
            return [[]]
        size = int(size) if size else len(files)
        return [files[i:i + size] for i in range(0, len(files), size)]

    # ---- FHiCL generation ---------------------------------------------------------------
    def assignments(self, params: Dict[str, Any]):
        return list(self.config["overrides"].items()) + self.space.assignments(params)

    def save_config(self):
        cfg = copy.deepcopy(self.config)
        cfg["study"].update(base_dir=str(self.base_dir), workdir=str(self.workdir))
        cfg["template"] = str(self._resolve(cfg["template"]))
        self.workdir.mkdir(parents=True, exist_ok=True)
        try:
            self.config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        except yaml.YAMLError:
            log.warning("config contains non-YAML objects (callables?); only in_process execution will work")

    def _make_point(self, pid: int, gen: int, x: np.ndarray) -> Point:
        params = self.space.decode(x)
        fcl = self.workdir / "fcls" / f"gen{gen:03d}" / f"point_{pid:06d}.fcl"
        self.template.write(self.assignments(params), fcl, tag=f"{self.name} point {pid}")
        rd = self.run_dir(pid)
        rd.mkdir(parents=True, exist_ok=True)
        (rd / "result.json").unlink(missing_ok=True)
        (rd / "point.json").write_text(json.dumps(
            {"id": pid, "generation": gen, "params": params, "fcl": str(fcl), "x": [float(v) for v in x]}))
        return Point(pid, gen, np.asarray(x, float), params, fcl, rd)

    def _points_from_rows(self, df: pd.DataFrame) -> List[Point]:
        out = []
        for _, r in df.sort_values("point_id").iterrows():
            x = np.array(json.loads(r["unit_x"]))
            pid = int(r["point_id"])
            out.append(Point(pid, int(r["generation"]), x, self.space.decode(x), Path(r["fcl"]), self.run_dir(pid)))
        return out

    # ---- optimizer synchronisation / resume -------------------------------------------------
    def _sync(self):
        """Replay stored history into the (deterministic) optimizer, once per Study object."""
        if self._synced:
            return
        self._synced = True
        df = self.writer.read_points()
        if df.empty:
            return
        for gen, g in df.groupby("generation", sort=True):
            g = g.sort_values("point_id")
            stored = np.array([json.loads(s) for s in g["unit_x"]])
            asked = self.optimizer.ask(len(g))
            if asked.shape != stored.shape or not np.allclose(asked, stored, atol=1e-9):
                log.warning("generation %d: optimizer replay differs from stored points "
                            "(config/seed changed?) - continuing with the stored points", gen)
            if (g["status"] == "pending").any():
                break
            self.optimizer.tell(stored, self._losses(g))

    def _penalty(self, finite: np.ndarray) -> float:
        p = self.config["failure"].get("penalty")
        if p is not None:
            return float(p)
        if len(finite) == 0:
            return 1e6
        return float(finite.max() + ((finite.max() - finite.min()) or 1.0))

    def _losses(self, g: pd.DataFrame) -> np.ndarray:
        def loss(df):
            obj = pd.to_numeric(df["objective"], errors="coerce").to_numpy(float)
            ok = (df["status"] == "done").to_numpy() & np.isfinite(obj)
            lo = np.where(ok, obj if self.direction == "minimize" else -obj, np.nan)
            return lo, ok
        lo, ok = loss(g)
        if ok.all():
            return lo
        allp = self.writer.read_points()
        alo, aok = loss(allp)
        pen = self._penalty(np.concatenate([alo[aok], lo[ok]]))
        return np.where(ok, lo, pen)

    # ---- generation lifecycle --------------------------------------------------------------------
    def _new_generation(self) -> List[Point]:
        X = self.optimizer.ask()
        if len(X) == 0:
            return []
        df = self.writer.read_points()
        pid0 = int(df["point_id"].max()) + 1 if not df.empty else 0
        gen = int(df["generation"].max()) + 1 if not df.empty else 0
        points, rows = [], []
        for i, x in enumerate(X):
            p = self._make_point(pid0 + i, gen, x)
            points.append(p)
            rows.append({"point_id": p.id, "generation": gen, "status": "pending",
                         "timestamp": datetime.now().isoformat(timespec="seconds"), "fcl": str(p.fcl),
                         "run_dir": str(p.run_dir), "unit_x": json.dumps([float(v) for v in x]),
                         **self.space.flat(p.params), "objective": np.nan, "duration": np.nan,
                         "n_events": 0, "error": ""})
        self.writer.write_points(pd.DataFrame(rows))
        log.info("generation %d: %d new point(s) (ids %d-%d)", gen, len(points), points[0].id, points[-1].id)
        return points

    def _pending(self) -> List[Point]:
        df = self.writer.read_points()
        return self._points_from_rows(df[df["status"] == "pending"]) if not df.empty else []

    def _collect(self, points: List[Point]) -> int:
        cur = self.writer.read_points().set_index("point_id", drop=False)
        rows, frames = [], []
        for p in points:
            rf = p.run_dir / "result.json"
            if not rf.exists():
                continue
            res = json.loads(rf.read_text())
            row = cur.loc[p.id].to_dict()
            row.update(status=res["status"], duration=res.get("duration", np.nan),
                       error=(res.get("error") or "")[:2000], timestamp=datetime.now().isoformat(timespec="seconds"))
            events = None
            if res["status"] == "done":
                try:
                    events = pd.read_csv(p.run_dir / "events.csv")
                    vals = self.objective(events, p.params)
                    if not np.isfinite(vals["objective"]):
                        raise ValueError("objective is not finite")
                    row.update(vals)
                    row["n_events"] = len(events)
                except Exception as exc:  # noqa: BLE001
                    row.update(status="failed", error=f"objective evaluation failed: {exc}")
                    events = None
            rows.append(row)
            if events is not None:
                events.insert(0, "point_id", p.id)
                frames.append(events)
        if frames:
            self.writer.write_events(pd.concat(frames, ignore_index=True))
        if rows:
            self.writer.write_points(pd.DataFrame(rows).infer_objects())
        return len(rows)

    def _tell_generation(self, gen: int):
        df = self.writer.read_points()
        g = df[df["generation"] == gen].sort_values("point_id")
        X = np.array([json.loads(s) for s in g["unit_x"]])
        self.optimizer.tell(X, self._losses(g))

    # ---- public API --------------------------------------------------------------------------------
    def create(self) -> List[Point]:
        """Generate the next generation's FHiCLs (without running anything)."""
        self.save_config()
        self._sync()
        pending = self._pending()
        return pending or self._new_generation()

    def run(self, max_generations: Optional[int] = None) -> dict:
        """Run (or resume) the optimisation loop. Returns :meth:`status`.

        With an executor that does not block (grid without ``submit``/``wait``) the loop
        stops as soon as a generation is pending; call ``run()`` / ``collect()`` again later."""
        self.save_config()
        self._sync()
        n_new = 0
        while True:
            pending = self._pending()
            if not pending:
                if self.optimizer.done or (max_generations is not None and n_new >= max_generations):
                    break
                pending = self._new_generation()
                if not pending:
                    break
                n_new += 1
            todo = [p for p in pending if not (p.run_dir / "result.json").exists()]
            if todo:
                self.executor.launch(self, todo)
                self.executor.wait(self, todo)
            self._collect(pending)
            if self._pending():
                log.info("generation %d is not finished yet; re-run later to continue", pending[0].generation)
                break
            self._tell_generation(pending[0].generation)
        return self.status()

    def collect(self) -> dict:
        """Ingest finished point results (e.g. from grid jobs) into the results tables."""
        n = self._collect(self._pending())
        log.info("collected %d point(s)", n)
        return self.status()

    def status(self) -> dict:
        df = self.writer.read_points()
        if df.empty:
            return {"points": 0, "generations": 0, "done": 0, "failed": 0, "pending": 0, "best": None}
        counts = df["status"].value_counts().to_dict()
        best = self.best(1)
        return {"points": len(df), "generations": int(df["generation"].nunique()),
                "done": counts.get("done", 0), "failed": counts.get("failed", 0), "pending": counts.get("pending", 0),
                "best": None if best.empty else {"point_id": int(best.iloc[0]["point_id"]),
                                                 "objective": float(best.iloc[0]["objective"])}}

    # ---- results ------------------------------------------------------------------------------------
    def points(self) -> pd.DataFrame:
        return self.writer.read_points()

    def events(self) -> pd.DataFrame:
        return self.writer.read_events()

    def view(self) -> ResultsView:
        return ResultsView(self.points(), self.space.flat_names(), events_loader=self.events,
                           direction=self.direction, summary_columns=list(self.config["summaries"]), name=self.name)

    def merged(self) -> pd.DataFrame:
        return self.view().merged()

    def best(self, n: int = 1) -> pd.DataFrame:
        return self.view().best(n)

    def reduce(self, conf: dict, aggregate: bool = True) -> pd.DataFrame:
        """Filter/aggregate the event-level table with a ``merge``-style config dict."""
        return reduce_frame(self.merged(), conf, key=["point_id"], aggregate=aggregate,
                            keep_first=self.space.flat_names() + ["objective", "generation"])

    def plot(self, name: str, save: Optional[str] = None, **kwargs):
        """Draw a registered plot, e.g. ``study.plot("slices", columns=["chi2NDF"])``."""
        out = self.workdir / "plots"
        out.mkdir(parents=True, exist_ok=True)
        return PLOTS.create(name, base_dir=out, **kwargs)(self.view(), save=save)

    def make_plots(self) -> List[Path]:
        paths = []
        for spec in self.config["plots"]:
            spec = dict(spec)
            name = spec.pop("name")
            spec.setdefault("output", f"{name}.png")
            self.plot(name, **spec)
            paths.append(self.workdir / "plots" / spec["output"])
        return paths

    # ---- validation ---------------------------------------------------------------------------------
    def validate(self, use_fhicl_dump: bool = False) -> List[str]:
        """Build everything, render the centre of the search space and report problems."""
        warnings: List[str] = []
        _ = (self.optimizer, self.executor, self.metric, self.objective, self.writer)
        assignments = self.assignments(self.space.decode(np.full(self.space.dims, 0.5)))
        try:
            files = self.input_files()
        except OSError as exc:
            files = []
            warnings.append(f"cannot read input list: {exc}")
        if not files:
            warnings.append("no input files configured")
        warnings += [f"input file not found: {f}" for f in files[:50] if "://" not in f and not Path(f).exists()]
        if use_fhicl_dump:
            try:
                warnings += check_assignments(assignments, flatten_dump(fhicl_dump(self.template.path)))
            except FclError as exc:
                warnings.append(f"fhicl-dump check skipped: {exc}")
        return warnings
