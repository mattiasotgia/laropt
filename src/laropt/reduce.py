"""Filter / transform / aggregate an event-level table (same YAML semantics as the old ``merge -r``)."""
from __future__ import annotations

from typing import Optional, Sequence

import pandas as pd


def reduce_frame(df: pd.DataFrame, conf: dict, key: Sequence[str] = ("point_id",), aggregate: bool = True,
                 keep_first: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """conf keys: ``filters`` (list of query strings), ``pre_transforms`` / ``post_transforms``
    (``{column: eval-formula}``), ``aggregation`` (``{method: [columns]}`` per ``key`` group).
    Columns in ``keep_first`` (e.g. the parameters) are carried through with ``first``."""
    key = list(key)
    df = df.copy()
    for flt in conf.get("filters", []):
        df = df.query(flt)
    for col, formula in conf.get("pre_transforms", {}).items():
        df[col] = df.eval(formula)
    out = df
    if aggregate and conf.get("aggregation"):
        agg = {c: m for m, cols in conf["aggregation"].items() for c in cols if c in df.columns and c not in key}
        for c in keep_first or []:
            if c in df.columns and c not in agg and c not in key:
                agg[c] = "first"
        out = df.groupby(key).agg(agg).reset_index()
    for col, formula in conf.get("post_transforms", {}).items():
        out[col] = out.eval(formula)
    return out
