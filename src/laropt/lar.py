"""Thin wrapper around the ``lar`` executable."""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Sequence


class LarError(RuntimeError):
    pass


def run_lar(
    fcl,
    inputs: Sequence[str] = (),
    output=None,
    tfile=None,
    n_events: int = -1,
    executable: str = "lar",
    extra_args: Sequence[str] = (),
    log=None,
    cwd=None,
    timeout=None,
    list_file=None,
):
    cmd = [executable, "-c", str(fcl)]
    inputs = [str(i) for i in inputs]
    if inputs:
        if len(inputs) == 1 and inputs[0].endswith(".root"):
            cmd += ["-s", inputs[0]]
        else:
            if list_file is None:
                raise ValueError("list_file is required for multiple inputs")
            Path(list_file).write_text("\n".join(inputs) + "\n")
            cmd += ["--source-list", str(list_file)]
    if output:
        cmd += ["-o", str(output)]
    if tfile:
        cmd += ["-T", str(tfile)]
    cmd += ["-n", str(n_events), *map(str, extra_args)]
    try:
        proc = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=cwd, timeout=timeout
        )
    except FileNotFoundError as exc:
        raise LarError(f"executable '{executable}' not found") from exc
    except subprocess.TimeoutExpired as exc:
        raise LarError(f"lar timed out after {timeout}s: {' '.join(cmd)}") from exc
    if log:
        Path(log).write_text(f"$ {' '.join(cmd)}\n{proc.stdout}")
    if proc.returncode != 0:
        tail = "\n".join(proc.stdout.splitlines()[-15:])
        raise LarError(f"lar failed with code {proc.returncode}: {' '.join(cmd)}\n{tail}")
    return proc
