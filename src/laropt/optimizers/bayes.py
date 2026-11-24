"""Batch Bayesian optimisation: Gaussian process + expected improvement (kriging-believer batches)."""
from __future__ import annotations

import warnings

import numpy as np

from ..registry import register
from .base import Optimizer


@register("optimizer", "bayes")
class BayesianOptimizer(Optimizer):
    """Generation 0 is an LHS design of ``n_initial`` points, then ``n_iterations``
    generations of ``batch_size`` points chosen by maximising the acquisition
    (``ei`` or ``lcb``) under a Matern-5/2 GP. Needs scikit-learn."""

    name = "bayes"

    def __init__(self, dims, seed=None, n_initial: int = 20, n_iterations: int = 10, batch_size: int = 5,
                 acquisition: str = "ei", xi: float = 0.01, kappa: float = 2.0,
                 n_candidates: int = 4000, n_restarts: int = 2):
        super().__init__(dims, seed)
        try:
            import sklearn  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ImportError("the 'bayes' optimizer needs scikit-learn (pip install laropt[bayes])") from exc
        from scipy.stats import qmc
        if acquisition not in ("ei", "lcb"):
            raise ValueError("acquisition must be 'ei' or 'lcb'")
        self.n_initial, self.n_iterations, self.batch_size = int(n_initial), int(n_iterations), int(batch_size)
        self.acquisition, self.xi, self.kappa = acquisition, xi, kappa
        self.n_candidates, self.n_restarts = int(n_candidates), int(n_restarts)
        self._initial = qmc.LatinHypercube(d=dims, seed=seed).random(self.n_initial)
        self._asked = 0
        self._told = 0

    @property
    def done(self):
        return self._asked >= 1 + self.n_iterations

    def _ask(self, n):
        if self._asked != self._told:
            raise RuntimeError("bayes: tell() the previous generation before ask()")
        self._asked += 1
        if self._asked == 1:
            return self._initial
        return self._suggest(n or self.batch_size)

    def _tell(self, X, y):
        self._told += 1

    # ------------------------------------------------------------------
    def _fit(self, X, y):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern, WhiteKernel
        kernel = (C(1.0, (1e-3, 1e3)) * Matern(length_scale=np.full(self.dims, 0.3),
                                                length_scale_bounds=(1e-2, 1e1), nu=2.5)
                  + WhiteKernel(1e-5, (1e-9, 1e-1)))
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=self.n_restarts,
                                      random_state=int(self.rng.integers(2**31 - 1)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp.fit(X, y)
        return gp

    def _candidates(self, X, y):
        n_local = self.n_candidates // 4
        cand = self.rng.random((self.n_candidates - n_local, self.dims))
        top = X[np.argsort(y)[:5]]
        local = top[self.rng.integers(len(top), size=n_local)] + self.rng.normal(0, 0.05, (n_local, self.dims))
        return np.clip(np.vstack([cand, local]), 0, 1)

    def _acq(self, gp, Xc, y):
        from scipy.stats import norm
        mu, sd = gp.predict(Xc, return_std=True)
        sd = np.maximum(sd, 1e-12)
        if self.acquisition == "ei":
            imp = y.min() - mu - self.xi * (np.std(y) or 1.0)
            z = imp / sd
            return imp * norm.cdf(z) + sd * norm.pdf(z)
        return -(mu - self.kappa * sd)

    def _suggest(self, n):
        X, y, out = self.X.copy(), self.y.copy(), []
        for _ in range(n):
            gp = self._fit(X, y)
            cand = self._candidates(X, y)
            x = cand[int(np.argmax(self._acq(gp, cand, y)))]
            out.append(x)
            X = np.vstack([X, x])
            y = np.append(y, gp.predict(x[None])[0])  # kriging believer
        return np.array(out)
