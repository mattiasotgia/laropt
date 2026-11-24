"""Executors: *where* the evaluation of a generation's points runs.

Every point is evaluated by the same worker entry point, ``laropt run-point``
(FHiCL -> ``lar`` -> metric -> ``result.json`` + ``events.csv`` in the point's
directory). Executors only decide how those commands are launched:

``local``  parallel subprocesses (or in-process, handy in notebooks / debugging).
``grid``   writes ``commands.txt`` + an array-job script and optionally submits and
           polls (``sbatch``, ``condor_submit``, ``jobsub_submit`` ... via the ``submit``
           template). Without ``submit`` it only prepares the files and the study
           pauses until the results appear (``laropt collect`` / ``laropt run`` again).
"""
from __future__ import annotations

import json
import logging
import shlex
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

from .base import Component
from .registry import register

log = logging.getLogger("laropt")

DEFAULT_COMMAND = "{python} -m laropt run-point -c {config} -p {point_id}"


class Executor(Component, ABC):
    name = "executor"

    def __init__(self, command: Optional[str] = None, setup: Optional[str] = None, base_dir=None):
        super().__init__(base_dir)
        self.command = command or DEFAULT_COMMAND
        self.setup = setup

    def worker_command(self, study, point) -> str:
        return self.command.format(python=shlex.quote(sys.executable), config=shlex.quote(str(study.config_path)),
                                   point_id=point.id, workdir=shlex.quote(str(study.workdir)))

    @abstractmethod
    def launch(self, study, points: List) -> None:
        """Start the evaluation of ``points`` (may block, may return immediately)."""

    def wait(self, study, points: List) -> bool:
        """Block until all ``points`` have a ``result.json``; return False if that did not happen."""
        return all((p.run_dir / "result.json").exists() for p in points)


@register("executor", "local")
class LocalExecutor(Executor):
    name = "local"

    def __init__(self, workers: int = 1, in_process: bool = False, timeout: Optional[float] = None,
                 command=None, setup=None, base_dir=None):
        super().__init__(command, setup, base_dir)
        self.workers, self.in_process, self.timeout = int(workers), in_process, timeout

    def _one(self, study, p):
        if self.in_process:
            from .worker import run_point
            run_point(study, p.id)
            return
        cmd = self.worker_command(study, p)
        if self.setup:
            cmd = f"{self.setup}\n{cmd}"
        with open(p.run_dir / "worker.log", "w") as logf:
            try:
                rc = subprocess.run(cmd, shell=True, executable="/bin/bash", stdout=logf, stderr=subprocess.STDOUT,
                                    cwd=p.run_dir, timeout=self.timeout).returncode
                msg = f"worker exited with code {rc}, see {p.run_dir / 'worker.log'}"
            except subprocess.TimeoutExpired:
                rc, msg = -1, f"worker timed out after {self.timeout}s"
        if not (p.run_dir / "result.json").exists():  # crashed hard (segfault, timeout, ...)
            (p.run_dir / "result.json").write_text(json.dumps(
                {"point_id": p.id, "status": "failed", "error": msg, "duration": float("nan")}))
        log.info("point %d finished (%s)", p.id, json.loads((p.run_dir / "result.json").read_text())["status"])

    def launch(self, study, points):
        log.info("running %d point(s) with %d local worker(s)", len(points), self.workers)
        with ThreadPoolExecutor(max_workers=max(1, self.workers)) as pool:
            list(pool.map(lambda p: self._one(study, p), points))


JOB_SCRIPT = """#!/bin/bash
@HEADER@
# one array task = one line of commands.txt (0-based task id)
IDX=${LAROPT_TASK_ID:-${SLURM_ARRAY_TASK_ID:-${PROCESS:-${1:-0}}}}
@SETUP@
CMD=$(sed -n "$((IDX+1))p" "@COMMANDS@")
echo "[laropt] task $IDX: $CMD"
eval "$CMD"
"""


@register("executor", "grid")
class GridExecutor(Executor):
    """
    submit     shell template run once per generation, with ``{script} {commands} {n_jobs} {n_last}
               {jobdir} {workdir} {generation}`` e.g. ``sbatch --array=0-{n_last} {script}``.
    header     lines inserted at the top of the job script (``#SBATCH ...`` / ``#$ ...``).
    wait       poll for the results after submitting (``poll_interval`` seconds, up to ``timeout``).
    """
    name = "grid"

    def __init__(self, submit: Optional[str] = None, header: str = "", wait: bool = True,
                 poll_interval: float = 30.0, timeout: Optional[float] = None,
                 command=None, setup=None, base_dir=None):
        super().__init__(command, setup, base_dir)
        self.submit, self.header = submit, header
        self.wait_for_jobs, self.poll_interval, self.timeout = wait, float(poll_interval), timeout
        self._submitted = False

    def launch(self, study, points):
        gen = points[0].generation
        jobdir = study.workdir / "jobs" / f"gen{gen:03d}"
        jobdir.mkdir(parents=True, exist_ok=True)
        commands = jobdir / "commands.txt"
        commands.write_text("\n".join(self.worker_command(study, p) for p in points) + "\n")
        script = jobdir / "job.sh"
        script.write_text(JOB_SCRIPT.replace("@HEADER@", self.header).replace("@SETUP@", self.setup or "")
                          .replace("@COMMANDS@", str(commands)))
        script.chmod(0o755)
        self._submitted = False
        if not self.submit:
            log.info("wrote %d job(s) to %s; submit them yourself (array task id 0-%d)", len(points), jobdir,
                     len(points) - 1)
            return
        cmd = self.submit.format(script=script, commands=commands, n_jobs=len(points), n_last=len(points) - 1,
                                 jobdir=jobdir, workdir=study.workdir, generation=gen)
        log.info("submitting: %s", cmd)
        subprocess.run(cmd, shell=True, check=True, cwd=jobdir)
        self._submitted = True

    def wait(self, study, points):
        if not self._submitted or not self.wait_for_jobs:
            return super().wait(study, points)
        start = time.time()
        while not super().wait(study, points):
            if self.timeout and time.time() - start > self.timeout:
                log.warning("timed out waiting for grid jobs")
                return False
            time.sleep(self.poll_interval)
        return True
