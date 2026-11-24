"""Read-only view over a study's tables, used by plots and analysis helpers."""
from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np
import pandas as pd


class ResultsView:
    def __init__(self, points: pd.DataFrame, param_columns: List[str], events: Optional[pd.DataFrame] = None,
                 events_loader: Optional[Callable[[], pd.DataFrame]] = None, direction: str = "minimize",
                 summary_columns: Optional[List[str]] = None, name: str = "study"):
        self.points = points
        self.param_columns = list(param_columns)
        self._events = events
        self._events_loader = events_loader
        self.direction = direction
        self.summary_columns = list(summary_columns or [])
        self.name = name

    @property
    def events(self) -> pd.DataFrame:
        if self._events is None:
            self._events = self._events_loader() if self._events_loader else pd.DataFrame()
        return self._events

    @property
    def ok(self) -> pd.DataFrame:
        """Successfully evaluated points, in evaluation order."""
        df = self.points[self.points["status"] == "done"] if "status" in self.points else self.points
        return df.sort_values("point_id").reset_index(drop=True)

    def running_best(self) -> np.ndarray:
        y = self.ok["objective"].to_numpy(dtype=float)
        return np.minimum.accumulate(y) if self.direction == "minimize" else np.maximum.accumulate(y)

    def best(self, n: int = 1) -> pd.DataFrame:
        return self.ok.sort_values("objective", ascending=self.direction == "minimize").head(n)

    def merged(self) -> pd.DataFrame:
        """Events joined with their point's parameters and objective (the old one-big-table layout)."""
        keep = [c for c in ["point_id", "generation", "objective", *self.summary_columns, *self.param_columns]
                if c in self.points.columns]
        return self.events.merge(self.points[keep], on="point_id", how="left")
