"""Search-space definition: how unit-cube coordinates map to FHiCL parameter values.

Optimizers only ever see the unit hypercube ``[0, 1]^d``. A :class:`ParamSpace`
decodes a point of that cube into named parameter values and into the list of
``(fhicl.path, value)`` assignments that are appended to the template FHiCL.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def fcl_literal(v: Any) -> str:
    """Render a Python value as a FHiCL literal. Strings are quoted unless prefixed with ``raw:``."""
    if isinstance(v, (list, tuple, np.ndarray)):
        return "[" + ", ".join(fcl_literal(x) for x in v) + "]"
    if isinstance(v, (bool, np.bool_)):
        return "true" if v else "false"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return repr(float(v))
    if isinstance(v, str):
        if v.startswith("raw:"):
            return v[4:]
        return '"' + v.replace('"', '\\"') + '"'
    return str(v)


@dataclass
class Param:
    """One tunable quantity, possibly a vector, possibly written to many FHiCL paths.

    paths        FHiCL dotted paths receiving the value. May contain ``{i}`` (vector element,
                 with ``per_element: true``) and any key of ``expand`` (e.g. ``{producer}``).
    size         vector length (1 = scalar).
    tied         if True, a vector uses one search dimension and all elements are equal.
    per_element  write each element as a scalar to ``path.format(i=i)`` instead of the
                 whole vector to ``path``.
    expand       ``{name: [values]}``: every path is written once per combination.
    """

    name: str
    kind: str = "float"  # float | int | choice
    bounds: Optional[Tuple[float, float]] = None
    choices: Optional[list] = None
    size: int = 1
    tied: bool = False
    log: bool = False
    decimals: Optional[int] = None
    paths: List[str] = field(default_factory=list)
    per_element: bool = False
    expand: Dict[str, list] = field(default_factory=dict)

    def __post_init__(self):
        if self.kind not in ("float", "int", "choice"):
            raise ValueError(f"{self.name}: kind must be float|int|choice, got '{self.kind}'")
        if self.kind == "choice":
            if not self.choices:
                raise ValueError(f"{self.name}: choice parameter needs 'choices'")
        else:
            if not self.bounds or len(self.bounds) != 2:
                raise ValueError(f"{self.name}: needs bounds [low, high]")
            lo, hi = self.bounds
            if (self.kind == "float" and not lo < hi) or (self.kind == "int" and not lo <= hi):
                raise ValueError(f"{self.name}: invalid bounds {self.bounds}")
            if self.log and lo <= 0:
                raise ValueError(f"{self.name}: log scale needs positive bounds")
        if self.size < 1:
            raise ValueError(f"{self.name}: size must be >= 1")
        if not self.paths:
            raise ValueError(f"{self.name}: needs at least one FHiCL path in 'paths'")
        if self.per_element and not all("{i}" in p for p in self.paths):
            raise ValueError(f"{self.name}: per_element requires '{{i}}' in every path")

    # -- dimensionality -------------------------------------------------
    @property
    def n_dims(self) -> int:
        return 1 if (self.size == 1 or self.tied) else self.size

    def flat_names(self) -> List[str]:
        return [self.name] if self.size == 1 else [f"{self.name}_{i}" for i in range(self.size)]

    # -- unit <-> value ---------------------------------------------------
    def _decode1(self, u: float):
        u = min(max(float(u), 0.0), 1.0)
        if self.kind == "choice":
            n = len(self.choices)
            return self.choices[min(int(u * n), n - 1)]
        lo, hi = self.bounds
        if self.kind == "int":
            n = int(hi) - int(lo) + 1
            return int(lo) + min(int(u * n), n - 1)
        if self.log:
            v = math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))
        else:
            v = lo + u * (hi - lo)
        v = round(v, self.decimals) if self.decimals is not None else float(f"{v:.6g}")
        return float(v)

    def _encode1(self, v) -> float:
        if self.kind == "choice":
            n = len(self.choices)
            return (self.choices.index(v) + 0.5) / n
        lo, hi = self.bounds
        if self.kind == "int":
            n = int(hi) - int(lo) + 1
            return (int(v) - int(lo) + 0.5) / n
        if self.log:
            u = (math.log(v) - math.log(lo)) / (math.log(hi) - math.log(lo))
        else:
            u = (v - lo) / (hi - lo)
        return min(max(u, 0.0), 1.0)

    def decode(self, u: Sequence[float]):
        vals = [self._decode1(x) for x in u]
        if self.size == 1:
            return vals[0]
        return [vals[0]] * self.size if self.tied else vals

    def encode(self, value) -> np.ndarray:
        vals = list(value) if isinstance(value, (list, tuple, np.ndarray)) else [value]
        if self.tied:
            vals = vals[:1]
        return np.array([self._encode1(v) for v in vals], dtype=float)

    def flat(self, value) -> Dict[str, Any]:
        vals = value if isinstance(value, list) else [value]
        return dict(zip(self.flat_names(), vals))

    # -- FHiCL assignments --------------------------------------------------
    def _expansions(self) -> List[Dict[str, Any]]:
        if not self.expand:
            return [{}]
        keys = list(self.expand)
        return [dict(zip(keys, combo)) for combo in itertools.product(*(self.expand[k] for k in keys))]

    def assignments(self, value) -> List[Tuple[str, Any]]:
        out = []
        for combo in self._expansions():
            for path in self.paths:
                if self.per_element:
                    vals = value if isinstance(value, list) else [value]
                    for i, v in enumerate(vals):
                        out.append((path.format(i=i, **combo), v))
                else:
                    out.append((path.format(**combo) if combo else path, value))
        return out


class ParamSpace:
    def __init__(self, params: Sequence[Param]):
        self.params = list(params)
        if len({p.name for p in self.params}) != len(self.params):
            raise ValueError("duplicate parameter names")
        self._offsets = np.cumsum([0] + [p.n_dims for p in self.params])

    @classmethod
    def from_config(cls, cfg: Dict[str, dict]) -> "ParamSpace":
        params = []
        for name, spec in cfg.items():
            s = dict(spec)
            if "type" in s:
                s["kind"] = s.pop("type")
            if "low" in s and "high" in s:
                s["bounds"] = (s.pop("low"), s.pop("high"))
            if "choices" in s:
                s["kind"] = "choice"
            for alias in ("path", "fcl_path", "fcl_paths"):
                if alias in s:
                    v = s.pop(alias)
                    s["paths"] = [v] if isinstance(v, str) else list(v)
            if s.get("bounds") is not None:
                s["bounds"] = tuple(s["bounds"])
            params.append(Param(name=name, **s))
        return cls(params)

    @property
    def dims(self) -> int:
        return int(self._offsets[-1])

    @property
    def names(self) -> List[str]:
        return [p.name for p in self.params]

    def flat_names(self) -> List[str]:
        return [n for p in self.params for n in p.flat_names()]

    def decode(self, x: Sequence[float]) -> Dict[str, Any]:
        x = np.asarray(x, dtype=float).ravel()
        if len(x) != self.dims:
            raise ValueError(f"expected {self.dims} coordinates, got {len(x)}")
        return {p.name: p.decode(x[a:b]) for p, a, b in zip(self.params, self._offsets[:-1], self._offsets[1:])}

    def encode(self, values: Dict[str, Any]) -> np.ndarray:
        return np.concatenate([p.encode(values[p.name]) for p in self.params])

    def flat(self, values: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for p in self.params:
            out.update(p.flat(values[p.name]))
        return out

    def assignments(self, values: Dict[str, Any]) -> List[Tuple[str, Any]]:
        out: List[Tuple[str, Any]] = []
        for p in self.params:
            out.extend(p.assignments(values[p.name]))
        return out

    def __len__(self):
        return len(self.params)

    def __str__(self):
        lines = ["Parameter space:"]
        for p in self.params:
            rng = f"choices={p.choices}" if p.kind == "choice" else f"bounds={list(p.bounds)}"
            extra = (", tied" if p.tied and p.size > 1 else "") + (", log" if p.log else "")
            lines.append(f"  - {p.name} ({p.kind}, size={p.size}{extra}, {rng}, {len(p.paths) * len(p._expansions())} path(s))")
        lines.append(f"  total search dimensions: {self.dims}")
        return "\n".join(lines)
