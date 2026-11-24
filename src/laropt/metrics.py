"""Metrics: turn the output of a tuned ``lar`` job into a per-event table.

Three flavours are provided:

``root_macro``    a C/C++ ROOT macro (compiled with ACLiC) called as ``function(lar_output, hist_file)``
                  returning ``vector<vector<double>>`` (one row per event) — like the old ``hitScorer.C``.
``fcl_analyzer``  a second FHiCL (analyzer / ntuple maker) run with ``lar`` on the tuned output; its
                  TFileService TTree (or CSV/JSON/parquet file) is read back as the table.
``python``        any Python callable ``f(ctx) -> DataFrame``.
"""
from __future__ import annotations

import json
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import pandas as pd

from .base import Component
from .fcl import FclTemplate
from .lar import run_lar
from .registry import load_callable, register


class MetricError(RuntimeError):
    pass


@dataclass
class MetricContext:
    """Everything a metric may need to know about the job it scores."""
    point_id: int
    point_dir: Path
    fcl: Path
    lar_output: Optional[Path]
    hist_file: Path
    input_files: List[str]
    batch: int
    params: Dict[str, Any]
    lar: Dict[str, Any] = field(default_factory=dict)  # the ``lar:`` config section

    def fields(self) -> Dict[str, str]:
        return {"point_id": str(self.point_id), "point_dir": str(self.point_dir), "fcl": str(self.fcl),
                "lar_output": str(self.lar_output or ""), "hist_file": str(self.hist_file),
                "batch": str(self.batch)}


class Metric(Component, ABC):
    name = "metric"

    @abstractmethod
    def evaluate(self, ctx: MetricContext) -> pd.DataFrame:
        """Return one row per event (or per whatever unit the objective aggregates)."""


def _to_frame(obj) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, dict):
        return pd.DataFrame(obj if all(hasattr(v, "__len__") and not isinstance(v, str) for v in obj.values())
                            else [obj])
    return pd.DataFrame(obj)


@register("metric", "python")
class PythonMetric(Metric):
    name = "python"

    def __init__(self, function: Union[str, Callable], base_dir=None):
        super().__init__(base_dir)
        self.fn = load_callable(function, self.base_dir)

    def evaluate(self, ctx):
        return _to_frame(self.fn(ctx))


@register("metric", "root_macro")
class RootMacroMetric(Metric):
    """C-style ROOT macro metric.

    columns   names (or ``{name, dtype}``) of the values of each returned row.
    function  entry-point name (default: macro file stem).
    args      call arguments, formatted with ``{lar_output} {hist_file} {point_dir} {fcl} {batch}``.
    isolate   copy/compile the macro inside the point directory so parallel jobs never
              race on the same ``*_C.so`` (recommended).
    headers   extra ``#include``s to load before compiling (LArSoft gallery headers by default).
    """
    name = "root_macro"
    DEFAULT_HEADERS = ["gallery/Event.h", "canvas/Persistency/Common/FindManyP.h", "canvas/Utilities/InputTag.h"]

    def __init__(self, macro: str, columns: Sequence[Union[str, Dict[str, str]]], function: Optional[str] = None,
                 args: Optional[Sequence[str]] = None, isolate: bool = True, headers: Optional[Sequence[str]] = None,
                 include_paths: Sequence[str] = (), support_files: Sequence[str] = (), base_dir=None):
        super().__init__(base_dir)
        self.macro = self.resolve(macro)
        self.function = function or self.macro.stem
        self.args = list(args) if args is not None else ["{lar_output}", "{hist_file}"]
        self.isolate = isolate
        self.headers = list(self.DEFAULT_HEADERS if headers is None else headers)
        self.include_paths = [str(self.resolve(p)) for p in include_paths]
        self.support_files = [self.resolve(p) for p in support_files]
        self.columns, self.dtypes = [], {}
        for c in columns:
            if isinstance(c, dict):
                self.columns.append(c["name"])
                if c.get("dtype"):
                    self.dtypes[c["name"]] = {"INTEGER": "int64", "REAL": "float64"}.get(c["dtype"], c["dtype"])
            else:
                self.columns.append(c)
        self._loaded = False

    def _load(self, point_dir: Path):
        import ROOT  # imported lazily: only the worker needs it
        macro = self.macro
        if self.isolate:
            dst = point_dir / "macro"
            dst.mkdir(exist_ok=True)
            for f in [self.macro, *self.support_files]:
                shutil.copy(f, dst / f.name)
            macro = dst / self.macro.name
        for p in {str(self.macro.parent), str(macro.parent), *self.include_paths}:
            ROOT.gSystem.AddDynamicPath(p)
            ROOT.gInterpreter.AddIncludePath(p)
        for h in self.headers:
            ROOT.gInterpreter.ProcessLine(f'#include "{h}"')
        ROOT.gROOT.ProcessLine(f".L {macro}+")
        if not hasattr(ROOT, self.function):
            raise MetricError(f"function '{self.function}' not found after compiling {macro}")
        self._loaded = True

    def evaluate(self, ctx):
        import ROOT
        if not self._loaded:
            self._load(ctx.point_dir)
        fields = ctx.fields()
        result = getattr(ROOT, self.function)(*[a.format(**fields) for a in self.args])
        rows = [list(r) if hasattr(r, "__len__") else r for r in result]
        if rows and not isinstance(rows[0], list):
            rows = [rows]  # flat vector -> single row
        df = pd.DataFrame(rows, columns=self.columns)
        return df.astype(self.dtypes) if self.dtypes else df


@register("metric", "fcl_analyzer")
class FclAnalyzerMetric(Metric):
    """Run an analyzer FHiCL with ``lar`` and read its output back.

    fcl        analyzer FHiCL file.
    output     file the analyzer writes (``-T`` TFileService file), default ``metrics_b{batch}.root``.
               ``.root`` needs ``tree`` and uproot (or ROOT); ``.csv``/``.json``/``.parquet`` are read directly.
    overrides  ``{fhicl.path: value}`` appended to the analyzer FHiCL; string values may use
               ``{lar_output} {point_dir} {batch} ...``.
    source     ``lar_output`` (default: analyse the tuned job's output) or ``input`` (the original input files).
    """
    name = "fcl_analyzer"

    def __init__(self, fcl: str, output: str = "metrics_b{batch}.root", tree: Optional[str] = None,
                 overrides: Optional[Dict[str, Any]] = None, source: str = "lar_output",
                 extra_args: Sequence[str] = (), base_dir=None):
        super().__init__(base_dir)
        self.template = FclTemplate(self.resolve(fcl))
        self.output, self.tree = output, tree
        self.overrides = dict(overrides or {})
        if source not in ("lar_output", "input"):
            raise ValueError("source must be 'lar_output' or 'input'")
        self.source = source
        self.extra_args = list(extra_args)

    def evaluate(self, ctx):
        fields = ctx.fields()
        out = ctx.point_dir / self.output.format(**fields)
        assignments = [(k, v.format(**fields) if isinstance(v, str) else v) for k, v in self.overrides.items()]
        fcl = self.template.write(assignments, ctx.point_dir / f"analyzer_b{ctx.batch}.fcl")
        inputs = [str(ctx.lar_output)] if (self.source == "lar_output" and ctx.lar_output) else ctx.input_files
        run_lar(fcl, inputs, tfile=out, n_events=ctx.lar.get("n_events", -1),
                executable=ctx.lar.get("executable", "lar"), extra_args=self.extra_args,
                log=ctx.point_dir / f"analyzer_b{ctx.batch}.log", cwd=ctx.point_dir,
                timeout=ctx.lar.get("timeout"), list_file=ctx.point_dir / f"analyzer_in_b{ctx.batch}.list")
        return self._read(out)

    def _read(self, path: Path) -> pd.DataFrame:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix == ".json":
            return _to_frame(json.loads(path.read_text()))
        if suffix == ".parquet":
            return pd.read_parquet(path)
        if suffix == ".root":
            if not self.tree:
                raise MetricError("fcl_analyzer: 'tree' is required for .root output")
            try:
                import uproot
                with uproot.open(path) as f:
                    return f[self.tree].arrays(library="pd")
            except ImportError:
                import ROOT
                return pd.DataFrame(ROOT.RDataFrame(self.tree, str(path)).AsNumpy())
        raise MetricError(f"fcl_analyzer: unsupported output type '{suffix}'")
