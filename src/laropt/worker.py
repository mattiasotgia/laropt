"""Evaluation of a single grid point: tuned FHiCL -> ``lar`` -> metric -> shard files.

Writes ``events.csv`` and (last, atomically) ``result.json`` into the point directory;
it never touches the study database, so any number of workers can run concurrently
on any file system.
"""
from __future__ import annotations

import json
import logging
import os
import time
import traceback
from pathlib import Path

import pandas as pd

from .lar import run_lar
from .metrics import MetricContext

log = logging.getLogger("laropt")


def run_point(study, point_id: int) -> dict:
    pdir = study.run_dir(point_id)
    spec = json.loads((pdir / "point.json").read_text())
    lar = study.config["lar"]
    t0 = time.time()
    result = {"point_id": point_id, "status": "done", "error": ""}
    try:
        frames = []
        for b, files in enumerate(study.input_batches()):
            out = pdir / f"lar_out_b{b}.root" if lar.get("output", True) else None
            tfile = lar["tfile"].format(point_dir=pdir, batch=b) if lar.get("tfile") else None
            run_lar(spec["fcl"], files, output=out, tfile=tfile, n_events=lar.get("n_events", -1),
                    executable=lar.get("executable", "lar"), extra_args=lar.get("extra_args", []),
                    log=pdir / f"lar_b{b}.log", cwd=pdir, timeout=lar.get("timeout"),
                    list_file=pdir / f"input_b{b}.list")
            ctx = MetricContext(point_id=point_id, point_dir=pdir, fcl=Path(spec["fcl"]),
                                lar_output=out, hist_file=pdir / f"hist_b{b}.root", input_files=files, batch=b,
                                params=spec["params"], lar=lar)
            df = study.metric.evaluate(ctx)
            df = df.copy()
            df["batch"] = b
            frames.append(df)
            if out is not None and not lar.get("keep_output", False):
                out.unlink(missing_ok=True)
        events = pd.concat(frames, ignore_index=True)
        events.to_csv(pdir / "events.csv", index=False)
    except Exception as exc:  # noqa: BLE001 - every failure must become a result, never a crash
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-1500:]}")
        log.error("point %d failed: %s", point_id, exc)
    result["duration"] = time.time() - t0
    tmp = pdir / "result.json.tmp"
    tmp.write_text(json.dumps(result))
    os.replace(tmp, pdir / "result.json")
    return result
