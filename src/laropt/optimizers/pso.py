"""Synchronous particle swarm optimisation — one generation = one swarm iteration."""
from __future__ import annotations

import numpy as np

from ..registry import register
from .base import Optimizer


@register("optimizer", "pso")
class ParticleSwarm(Optimizer):
    name = "pso"

    def __init__(self, dims, seed=None, n_particles: int = 20, n_iterations: int = 10,
                 w: float = 0.72, c1: float = 1.49, c2: float = 1.49, vmax: float = 0.3):
        super().__init__(dims, seed)
        from scipy.stats import qmc
        self.n_particles, self.n_iterations = int(n_particles), int(n_iterations)
        self.w, self.c1, self.c2, self.vmax = w, c1, c2, vmax
        self.pos = qmc.LatinHypercube(d=dims, seed=seed).random(self.n_particles)
        self.vel = self.rng.uniform(-vmax, vmax, size=self.pos.shape) * 0.5
        self.pbest_x = self.pos.copy()
        self.pbest_y = np.full(self.n_particles, np.inf)
        self.gbest_x = self.pos[0].copy()
        self.gbest_y = np.inf
        self._asked = 0
        self._told = 0

    @property
    def done(self):
        return self._asked >= self.n_iterations

    def _ask(self, n):
        if self._asked != self._told:
            raise RuntimeError("PSO: tell() the previous generation before ask()")
        self._asked += 1
        return self.pos.copy()

    def _tell(self, X, y):
        if len(y) != self.n_particles:
            raise ValueError(f"PSO expects {self.n_particles} results per generation, got {len(y)}")
        X = np.clip(X, 0, 1)
        self.pos = X.copy()  # use what was actually evaluated (robust to replay)
        better = y < self.pbest_y
        self.pbest_x[better], self.pbest_y[better] = X[better], y[better]
        i = int(np.argmin(self.pbest_y))
        if self.pbest_y[i] < self.gbest_y:
            self.gbest_y, self.gbest_x = float(self.pbest_y[i]), self.pbest_x[i].copy()
        self._told += 1
        if self._told < self.n_iterations:
            r1 = self.rng.random(self.pos.shape)
            r2 = self.rng.random(self.pos.shape)
            self.vel = (self.w * self.vel + self.c1 * r1 * (self.pbest_x - self.pos)
                        + self.c2 * r2 * (self.gbest_x - self.pos))
            self.vel = np.clip(self.vel, -self.vmax, self.vmax)
            new = self.pos + self.vel
            out = (new < 0) | (new > 1)
            self.vel[out] *= -0.5  # bounce off the walls
            self.pos = np.clip(new, 0, 1)
