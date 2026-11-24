"""Optimizer interface (ask / tell, always on the unit hypercube, always minimising)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class Optimizer(ABC):
    """Ask/tell optimizer on ``[0, 1]^dims``.

    * ``ask(n=None)`` returns an ``(m, dims)`` array — one *generation* of points to
      evaluate (``m == 0`` when finished). ``n`` is a hint; fixed-size optimizers
      (e.g. PSO) ignore it.
    * ``tell(X, y)`` reports the losses (to be minimised) of a generation, in the
      same order as returned by ``ask``.

    Everything must be deterministic given ``seed``: laropt resumes a study by
    replaying the stored ask/tell history.
    """

    name = "optimizer"

    def __init__(self, dims: int, seed: Optional[int] = None):
        self.dims = int(dims)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.X = np.empty((0, self.dims))
        self.y = np.empty(0)

    @property
    @abstractmethod
    def done(self) -> bool: ...

    @abstractmethod
    def _ask(self, n: Optional[int]) -> np.ndarray: ...

    def _tell(self, X: np.ndarray, y: np.ndarray) -> None:
        pass

    def ask(self, n: Optional[int] = None) -> np.ndarray:
        if self.done:
            return np.empty((0, self.dims))
        return np.clip(np.atleast_2d(np.asarray(self._ask(n), dtype=float)), 0.0, 1.0)

    def tell(self, X, y) -> None:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).ravel()
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        self.X = np.vstack([self.X, X])
        self.y = np.concatenate([self.y, y])
        self._tell(X, y)

    @property
    def best(self):
        if not len(self.y):
            return None, None
        i = int(np.argmin(self.y))
        return self.X[i], float(self.y[i])
