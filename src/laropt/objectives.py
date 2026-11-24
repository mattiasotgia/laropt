"""Objectives: reduce the per-event table of one grid point to a single number."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .base import Component
from .registry import load_callable, register

_SAFE_BUILTINS = {"abs": abs, "min": min, "max": max, "sum": sum, "len": len,
                  "float": float, "int": int, "round": round, "pow": pow}


def eval_expression(expr: str, events: pd.DataFrame, params: Optional[dict] = None):
    """Evaluate ``expr`` with ``df`` (events), ``params``, ``np``, ``pd`` and every
    identifier-like column of ``df`` available as a name."""
    ns: Dict[str, Any] = {"__builtins__": _SAFE_BUILTINS, "np": np, "pd": pd, "df": events, "params": params or {}}
    for c in events.columns:
        if str(c).isidentifier() and c not in ns:
            ns[c] = events[c]
    return eval(expr, ns)  # noqa: S307 - expressions come from the user's own config


class Objective(Component, ABC):
    name = "objective"

    def __init__(self, direction: str = "minimize", summaries: Optional[Dict[str, str]] = None, base_dir=None):
        super().__init__(base_dir)
        if direction not in ("minimize", "maximize"):
            raise ValueError("direction must be 'minimize' or 'maximize'")
        self.direction = direction
        self.summaries = dict(summaries or {})

    @abstractmethod
    def value(self, events: pd.DataFrame, params: dict) -> float: ...

    def __call__(self, events: pd.DataFrame, params: dict) -> Dict[str, float]:
        out = {"objective": float(self.value(events, params))}
        for key, expr in self.summaries.items():
            try:
                out[key] = float(eval_expression(expr, events, params))
            except Exception:
                out[key] = float("nan")
        return out

    def loss(self, value):
        """Value as seen by the (always minimising) optimizers."""
        return value if self.direction == "minimize" else -value


@register("objective", "expression")
class ExpressionObjective(Objective):
    """``expr`` is evaluated on the events, e.g. ``"((plane2Ratio - 1)**2).mean()"``."""
    name = "expression"

    def __init__(self, expr: str, direction="minimize", summaries=None, base_dir=None):
        super().__init__(direction, summaries, base_dir)
        self.expr = expr

    def value(self, events, params):
        return eval_expression(self.expr, events, params)


@register("objective", "callable")
class CallableObjective(Objective):
    """``function`` is ``'module:func'`` / ``'file.py:func'`` / a callable ``f(events, params) -> float``."""
    name = "callable"

    def __init__(self, function, direction="minimize", summaries=None, base_dir=None):
        super().__init__(direction, summaries, base_dir)
        self.fn = load_callable(function, self.base_dir)

    def value(self, events, params):
        return self.fn(events, params)
