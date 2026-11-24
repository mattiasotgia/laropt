"""Test plugin: a synthetic metric that 'scores' the overrides found in the tuned FHiCL."""
import ast

import numpy as np
import pandas as pd

from laropt import register
from laropt.metrics import Metric


def read_overrides(fcl_text: str) -> dict:
    out = {}
    section = fcl_text.split("# laropt overrides", 1)[1]
    for line in section.splitlines()[1:]:  # first line is the rest of the header comment
        if ":" in line and not line.startswith("#"):
            k, v = line.split(":", 1)
            out[k.strip()] = ast.literal_eval(v.strip())
    return out


def loss_from_fcl(text: str) -> float:
    ov = read_overrides(text)
    w = [ov[f"physics.hf.vec[{i}]"] for i in range(3)]
    return (ov["physics.hf.Thr"] - 3.3) ** 2 + 0.5 * (ov["physics.hf.N"] - 4) ** 2 + sum((x - 0.5) ** 2 for x in w)


@register("metric", "synthetic")
class Synthetic(Metric):
    name = "synthetic"

    def evaluate(self, ctx):
        loss = loss_from_fcl(ctx.fcl.read_text())
        rng = np.random.default_rng(ctx.point_id)
        return pd.DataFrame({"event": range(5), "value": loss + 0.001 * rng.standard_normal(5)})
