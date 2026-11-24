"""Non-adaptive designs: Latin hypercube, Sobol, Halton, random, full grid."""
from __future__ import annotations

import itertools
import warnings
from typing import Optional, Sequence, Union

import numpy as np

from ..registry import register
from .base import Optimizer


class _Design(Optimizer):
    """One-shot design generated up front and handed out in slices (default: all at once)."""

    def __init__(self, dims, seed=None, batch_size: Optional[int] = None):
        super().__init__(dims, seed)
        self.batch_size = batch_size
        self._design = self._make()
        self._given = 0

    def _make(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def done(self):
        return self._given >= len(self._design)

    def _ask(self, n):
        n = n or self.batch_size or len(self._design)
        out = self._design[self._given:self._given + n]
        self._given += len(out)
        return out


@register("optimizer", "lhs")
class LatinHypercube(_Design):
    """Latin hypercube sampling (what the original tuning script used)."""
    name = "lhs"

    def __init__(self, dims, seed=None, n_samples: int = 200, batch_size=None):
        self.n_samples = int(n_samples)
        super().__init__(dims, seed, batch_size)

    def _make(self):
        from scipy.stats import qmc
        return qmc.LatinHypercube(d=self.dims, seed=self.seed).random(self.n_samples)


@register("optimizer", "sobol")
class Sobol(_Design):
    name = "sobol"

    def __init__(self, dims, seed=None, n_samples: int = 128, batch_size=None):
        self.n_samples = int(n_samples)
        super().__init__(dims, seed, batch_size)

    def _make(self):
        from scipy.stats import qmc
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return qmc.Sobol(d=self.dims, scramble=True, seed=self.seed).random(self.n_samples)


@register("optimizer", "halton")
class Halton(_Design):
    name = "halton"

    def __init__(self, dims, seed=None, n_samples: int = 128, batch_size=None):
        self.n_samples = int(n_samples)
        super().__init__(dims, seed, batch_size)

    def _make(self):
        from scipy.stats import qmc
        return qmc.Halton(d=self.dims, seed=self.seed).random(self.n_samples)


@register("optimizer", "random")
class RandomSearch(_Design):
    name = "random"

    def __init__(self, dims, seed=None, n_samples: int = 100, batch_size=None):
        self.n_samples = int(n_samples)
        super().__init__(dims, seed, batch_size)

    def _make(self):
        return self.rng.random((self.n_samples, self.dims))


@register("optimizer", "grid")
class FullGrid(_Design):
    """Full factorial grid at cell centres; ``levels`` is an int or one int per dimension."""
    name = "grid"

    def __init__(self, dims, seed=None, levels: Union[int, Sequence[int]] = 3, batch_size=None, max_points: int = 100000):
        self.levels = [int(levels)] * dims if isinstance(levels, int) else [int(l) for l in levels]
        if len(self.levels) != dims:
            raise ValueError(f"grid: need {dims} levels, got {len(self.levels)}")
        self.max_points = max_points
        super().__init__(dims, seed, batch_size)

    def _make(self):
        total = int(np.prod(self.levels))
        if total > self.max_points:
            raise ValueError(f"grid would have {total} points (> max_points={self.max_points})")
        axes = [(np.arange(L) + 0.5) / L for L in self.levels]
        return np.array(list(itertools.product(*axes)))
