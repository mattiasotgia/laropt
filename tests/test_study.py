import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from laropt import Study
from laropt.cli import cli
from synthetic_plugin import loss_from_fcl


def test_local_inprocess_lhs(make_config, env):
    s = Study(make_config(), base_dir=env)
    st = s.run()
    assert st["done"] == 12 and st["pending"] == 0 and st["generations"] == 1
    pts, ev = s.points(), s.events()
    assert len(ev) == 12 * 5 and {"w_0", "w_1", "w_2", "thr", "n", "objective", "vmax"} <= set(pts.columns)
    # FHiCL really contains the overrides, and objective matches the loss computed from them
    for _, r in pts.iterrows():
        assert abs(r.objective - loss_from_fcl(open(r.fcl).read())) < 0.01
    assert (env / "out" / "config.yaml").exists()
    assert not list((env / "out" / "runs").glob("*/lar_out*"))  # keep_output false


def test_subprocess_pso_parallel(make_config, env):
    cfg = make_config(optimizer={"name": "pso", "n_particles": 6, "n_iterations": 4},
                      executor={"name": "local", "workers": 3})
    s = Study(cfg, base_dir=env)
    st = s.run()
    assert st["done"] == 24 and st["generations"] == 4
    ids = s.points().sort_values("point_id")
    assert ids.groupby("generation").size().tolist() == [6] * 4
    # optimisation should improve on the first generation
    assert ids[ids.generation == 3].objective.min() <= ids[ids.generation == 0].objective.min()


def test_failures_become_penalties(make_config, env):
    cfg = make_config(overrides={"FAIL_ME": 1}, optimizer={"name": "pso", "n_particles": 4, "n_iterations": 2})
    st = Study(cfg, base_dir=env).run()
    assert st["failed"] == 8 and st["done"] == 0 and st["best"] is None


def test_maximize_direction(make_config, env):
    cfg = make_config(objective={"name": "expression", "expr": "df.value.mean()", "direction": "maximize"},
                      optimizer={"name": "pso", "n_particles": 6, "n_iterations": 3})
    s = Study(cfg, base_dir=env); s.run()
    assert s.best(1).objective.iloc[0] == s.points().objective.max()


def test_grid_pause_resume_matches_continuous(make_config, env, tmp_path):
    """grid executor without submit: study pauses each generation; resuming == uninterrupted run."""
    from laropt.worker import run_point
    base = dict(optimizer={"name": "pso", "n_particles": 5, "n_iterations": 3})
    ref = Study(make_config(**base, study={"name": "t", "workdir": "ref", "seed": 1}), base_dir=env)
    ref.run()

    cfg = make_config(**base, executor={"name": "grid"}, study={"name": "t", "workdir": "g", "seed": 1})
    for gen in range(3):
        s = Study(cfg, base_dir=env)            # fresh object each time = new process
        st = s.run()
        assert st["pending"] == 5 and st["generations"] == gen + 1
        assert (env / "g" / "jobs" / f"gen{gen:03d}" / "commands.txt").exists()
        for p in s._pending():                   # "the grid jobs run"
            run_point(s, p.id)
        s.collect()
    final = Study(cfg, base_dir=env).run()
    assert final["done"] == 15 and final["pending"] == 0
    a = ref.points().sort_values("point_id")[["thr", "n", "w_0", "w_1", "w_2"]].to_numpy(float)
    b = Study.open(env / "g").points().sort_values("point_id")[["thr", "n", "w_0", "w_1", "w_2"]].to_numpy(float)
    np.testing.assert_allclose(a, b)


def test_grid_submit_template_and_script(make_config, env):
    cfg = make_config(optimizer={"name": "random", "n_samples": 4},
                      executor={"name": "grid", "poll_interval": 0.1, "timeout": 60,
                                "submit": "for i in $(seq 0 {n_last}); do LAROPT_TASK_ID=$i bash {script}; done"},
                      study={"name": "t", "workdir": "gs", "seed": 1})
    st = Study(cfg, base_dir=env).run()
    assert st["done"] == 4


def test_fcl_analyzer_metric(make_config, env):
    cfg = make_config(metric={"name": "fcl_analyzer", "fcl": "ana.fcl", "output": "m_b{batch}.csv"},
                      optimizer={"name": "random", "n_samples": 3},
                      study={"name": "t", "workdir": "an", "seed": 1}, plugins=[])
    cfg["executor"] = {"name": "local", "workers": 2}
    cfg["lar"]["keep_output"] = True
    s = Study(cfg, base_dir=env)
    st = s.run()
    assert st["done"] == 3
    p = s.points().iloc[0]
    assert abs(p.objective - loss_from_fcl(open(p.fcl).read())) < 1e-9


def test_input_batches(make_config, env):
    (env / "files.list").write_text("\n".join(f"/d/f{i}.root" for i in range(5)))
    s = Study(make_config(input={"files": "files.list", "batch_size": 2}), base_dir=env)
    assert [len(b) for b in s.input_batches()] == [2, 2, 1]


def test_reduce_and_view(make_config, env):
    s = Study(make_config(), base_dir=env); s.run()
    out = s.reduce({"filters": ["event >= 1"], "pre_transforms": {"v2": "value * 2"},
                    "aggregation": {"mean": ["v2"], "count": ["event"]}, "post_transforms": {"r": "v2 / 2"}})
    assert len(out) == 12 and out["event"].eq(4).all()
    assert {"thr", "w_0", "objective"} <= set(out.columns)


def test_plots_and_cli(make_config, env, tmp_path):
    import matplotlib; matplotlib.use("Agg")
    import yaml
    cfg = make_config(plots=[{"name": "convergence"}, {"name": "slices", "output": "s.png"},
                             {"name": "parallel"}, {"name": "correlation"},
                             {"name": "scatter2d", "x": "thr", "y": "w_0"},
                             {"name": "events_hist", "column": "value"}])
    cfg["executor"] = {"name": "local", "workers": 2}
    yml = env / "study.yaml"
    yml.write_text(yaml.safe_dump(cfg))
    r = CliRunner()
    for args in (["validate", "-c", str(yml)], ["run", "-c", str(yml), "-s", "optimizer.n_samples=8"],
                 ["status", "-c", str(yml)], ["best", "-c", str(yml), "-n", "2"], ["plot", "-c", str(yml)],
                 ["list"]):
        res = r.invoke(cli, args)
        assert res.exit_code == 0, (args, res.output, res.exception)
    assert (env / "out" / "plots" / "convergence.png").exists() and (env / "out" / "plots" / "s.png").exists()
    # merge: study dir -> reduced pickle
    red = env / "red.yaml"
    red.write_text(yaml.safe_dump({"aggregation": {"mean": ["value"]}}))
    res = r.invoke(cli, ["merge", "-i", str(env / "out"), "-o", str(env / "m.pkl"), "-r", str(red)])
    assert res.exit_code == 0, res.output
    assert len(pd.read_pickle(env / "m.pkl")) == 8


@pytest.mark.parametrize("writer", ["csv", "pickle"])
def test_other_writers(make_config, env, writer):
    s = Study(make_config(writer={"name": writer}, optimizer={"name": "random", "n_samples": 3}), base_dir=env)
    assert s.run()["done"] == 3 and len(s.events()) == 15
