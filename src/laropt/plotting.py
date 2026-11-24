"""Plot classes. Each takes a :class:`ResultsView` and returns a matplotlib Figure.

In a notebook: ``study.plot("convergence")`` or ``PLOTS.create("slices")(view)``.
From YAML: ``plots: [{name: convergence, output: convergence.png}]``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .base import Component
from .registry import register
from .results import ResultsView


def _plt():
    import matplotlib.pyplot as plt
    return plt


class Plot(Component, ABC):
    name = "plot"

    def __init__(self, output: Optional[str] = None, title: Optional[str] = None, figsize=None, base_dir=None):
        super().__init__(base_dir)
        self.output, self.title, self.figsize = output, title, figsize

    @abstractmethod
    def draw(self, view: ResultsView):
        """Build and return a matplotlib Figure."""

    def __call__(self, view: ResultsView, save: Optional[str] = None):
        fig = self.draw(view)
        if self.title:
            fig.suptitle(self.title)
        target = save or self.output
        if target:
            path = self.resolve(target)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=150, bbox_inches="tight")
        return fig


@register("plot", "convergence")
class Convergence(Plot):
    """Objective of every evaluation (coloured by generation) and the best value so far."""
    name = "convergence"

    def draw(self, view):
        plt = _plt()
        fig, ax = plt.subplots(figsize=self.figsize or (7, 4))
        df = view.ok
        sc = ax.scatter(np.arange(len(df)), df["objective"], c=df["generation"], cmap="viridis", s=16)
        ax.plot(view.running_best(), color="k", lw=1.5, label="best so far")
        ax.set_xlabel("evaluation"); ax.set_ylabel("objective"); ax.legend()
        fig.colorbar(sc, ax=ax, label="generation")
        return fig


@register("plot", "slices")
class Slices(Plot):
    """Objective against each parameter (one panel per parameter column); best 10% highlighted."""
    name = "slices"

    def __init__(self, columns: Optional[Sequence[str]] = None, ncols: int = 3, **kw):
        super().__init__(**kw)
        self.columns, self.ncols = columns, ncols

    def draw(self, view):
        plt = _plt()
        cols = list(self.columns or view.param_columns)
        nrows = int(np.ceil(len(cols) / self.ncols))
        fig, axes = plt.subplots(nrows, self.ncols, figsize=self.figsize or (4 * self.ncols, 3 * nrows),
                                 squeeze=False, sharey=True)
        df = view.ok
        top = df.nsmallest(max(1, len(df) // 10), "objective") if view.direction == "minimize" \
            else df.nlargest(max(1, len(df) // 10), "objective")
        for ax, c in zip(axes.ravel(), cols):
            ax.scatter(df[c], df["objective"], s=8, alpha=0.5)
            ax.scatter(top[c], top["objective"], s=14, color="C3")
            ax.set_xlabel(c)
        for ax in axes.ravel()[len(cols):]:
            ax.axis("off")
        for ax in axes[:, 0]:
            ax.set_ylabel("objective")
        fig.tight_layout()
        return fig


@register("plot", "parallel")
class ParallelCoordinates(Plot):
    """One line per evaluation across min-max normalised parameters, coloured by objective."""
    name = "parallel"

    def __init__(self, top: int = 5, **kw):
        super().__init__(**kw)
        self.top = top

    def draw(self, view):
        plt = _plt()
        fig, ax = plt.subplots(figsize=self.figsize or (max(6, 1.1 * len(view.param_columns)), 4))
        df = view.ok
        cols = view.param_columns
        span = (df[cols].max() - df[cols].min()).replace(0, 1)
        norm = (df[cols] - df[cols].min()) / span
        cmap = plt.get_cmap("viridis_r" if view.direction == "minimize" else "viridis")
        o = df["objective"]
        colors = cmap((o - o.min()) / ((o.max() - o.min()) or 1))
        for i in range(len(df)):
            ax.plot(range(len(cols)), norm.iloc[i], color=colors[i], alpha=0.35, lw=0.8)
        best_idx = df.sort_values("objective", ascending=view.direction == "minimize").head(self.top).index
        for i in best_idx:
            ax.plot(range(len(cols)), norm.loc[i], color="k", lw=1.8)
        ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=45, ha="right")
        ax.set_ylabel("normalised value")
        return fig


@register("plot", "correlation")
class Correlation(Plot):
    """Spearman correlation between parameters and the objective / summaries."""
    name = "correlation"

    def draw(self, view):
        plt = _plt()
        df = view.ok
        cols = [*view.param_columns, "objective", *[c for c in view.summary_columns if c in df.columns]]
        corr = df[cols].corr(method="spearman")
        fig, ax = plt.subplots(figsize=self.figsize or (0.6 * len(cols) + 3, 0.6 * len(cols) + 2))
        im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=60, ha="right")
        ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols)
        fig.colorbar(im, ax=ax, label="Spearman rho")
        return fig


@register("plot", "scatter2d")
class Scatter2D(Plot):
    """Two parameters on the axes, objective as colour."""
    name = "scatter2d"

    def __init__(self, x: str, y: str, **kw):
        super().__init__(**kw)
        self.x, self.y = x, y

    def draw(self, view):
        plt = _plt()
        fig, ax = plt.subplots(figsize=self.figsize or (6, 5))
        df = view.ok
        sc = ax.scatter(df[self.x], df[self.y], c=df["objective"], cmap="viridis", s=24)
        ax.set_xlabel(self.x); ax.set_ylabel(self.y)
        fig.colorbar(sc, ax=ax, label="objective")
        return fig


@register("plot", "events_hist")
class EventsHistogram(Plot):
    """Distribution of an event-level column: all points vs the ``top`` best points."""
    name = "events_hist"

    def __init__(self, column: str, top: int = 3, bins: int = 50, **kw):
        super().__init__(**kw)
        self.column, self.top, self.bins = column, top, bins

    def draw(self, view):
        plt = _plt()
        ev = view.events
        best_ids = view.best(self.top)["point_id"].tolist()
        fig, ax = plt.subplots(figsize=self.figsize or (6, 4))
        lo, hi = ev[self.column].quantile([0.005, 0.995])
        rng = (lo, hi)
        ax.hist(ev[self.column], bins=self.bins, range=rng, density=True, alpha=0.4, label="all points")
        ax.hist(ev[ev["point_id"].isin(best_ids)][self.column], bins=self.bins, range=rng, density=True,
                histtype="step", lw=1.8, color="C3", label=f"best {len(best_ids)}")
        ax.set_xlabel(self.column); ax.legend()
        return fig
