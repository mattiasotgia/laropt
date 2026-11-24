import numpy as np
import pytest

from laropt import OPTIMIZERS, FclTemplate, Param, ParamSpace, fcl_literal, register
from laropt.fcl import check_assignments, flatten_dump


def test_param_roundtrip_and_assignments():
    sp = ParamSpace.from_config({
        "a": {"bounds": [3, 7], "size": 3, "per_element": True, "expand": {"p": ["X", "Y"]},
              "paths": ["t.{p}.Plane{i}.Roi"]},
        "b": {"type": "int", "bounds": [5, 10], "paths": ["t.B"]},
        "c": {"bounds": [1, 100], "log": True, "size": 3, "tied": True, "paths": ["t.C"]},
        "d": {"choices": ["u", "v"], "paths": ["t.D"]},
    })
    assert sp.dims == 3 + 1 + 1 + 1
    x = np.random.default_rng(0).random(sp.dims)
    vals = sp.decode(x)
    assert len(vals["a"]) == 3 and vals["c"][0] == vals["c"][1] == vals["c"][2]
    assert set(vals["b"] for _ in [0]) <= set(range(5, 11))
    np.testing.assert_allclose(sp.decode(sp.encode(vals)) == vals, True)
    paths = [p for p, _ in sp.assignments(vals)]
    assert "t.X.Plane2.Roi" in paths and "t.Y.Plane0.Roi" in paths and len(paths) == 6 + 1 + 1 + 1
    assert list(sp.flat(vals)) == ["a_0", "a_1", "a_2", "b", "c_0", "c_1", "c_2", "d"]


def test_int_sampling_is_uniform():
    p = Param("n", "int", bounds=(5, 10), paths=["x"])
    vals = [p.decode([u]) for u in np.linspace(0, 1, 6000)]
    counts = np.bincount(vals)[5:]
    assert counts.min() > 900 and p.decode([1.0]) == 10


def test_invalid_params():
    with pytest.raises(ValueError):
        Param("x", bounds=(2, 1), paths=["a"])
    with pytest.raises(ValueError):
        Param("x", bounds=(0, 1), per_element=True, paths=["a"])


def test_fcl_literal_and_render(tmp_path):
    assert fcl_literal([1, 2.5, True, "s"]) == '[1, 2.5, true, "s"]'
    assert fcl_literal("raw:@local::x") == "@local::x"
    f = tmp_path / "t.fcl"
    f.write_text("a: 1\n")
    out = FclTemplate(f).render([("a.b", [1, 2])], tag="t")
    assert out.startswith("a: 1") and out.rstrip().endswith("a.b: [1, 2]")


def test_flatten_dump():
    keys = flatten_dump("physics: {\n producers: {\n  g: {\n   Thr: 1\n   v: [ 1,\n 2 ]\n  }\n }\n}\n")
    assert {"physics.producers.g.Thr", "physics.producers.g.v"} <= keys
    w = check_assignments([("physics.producers.g.Thr", 1), ("physics.producers.g.New", 1), ("x.y", 1)], keys)
    assert len(w) == 2


def test_registry_custom_optimizer():
    from laropt.optimizers import Optimizer

    @register("optimizer", "unit-test-opt")
    class Mine(Optimizer):
        done = False
        def _ask(self, n): return np.zeros((1, self.dims))

    assert "unit-test-opt" in OPTIMIZERS
    assert OPTIMIZERS.create("unit-test-opt", dims=2).ask().shape == (1, 2)
    with pytest.raises(KeyError):
        OPTIMIZERS.get("nope")


def _sphere(x):
    return float(np.sum((x - 0.3) ** 2))


@pytest.mark.parametrize("name,kw,expected_gens", [
    ("lhs", {"n_samples": 30}, 1), ("sobol", {"n_samples": 16}, 1), ("halton", {"n_samples": 10}, 1),
    ("random", {"n_samples": 10}, 1), ("grid", {"levels": 3}, 1),
    ("pso", {"n_particles": 10, "n_iterations": 6}, 6),
    ("bayes", {"n_initial": 8, "n_iterations": 3, "batch_size": 2, "n_candidates": 500, "n_restarts": 0}, 4),
])
def test_optimizers_ask_tell(name, kw, expected_gens):
    opt = OPTIMIZERS.create(name, dims=3, seed=3, **kw)
    gens = 0
    while not opt.done:
        X = opt.ask()
        assert X.ndim == 2 and X.shape[1] == 3 and (X >= 0).all() and (X <= 1).all()
        opt.tell(X, [_sphere(x) for x in X])
        gens += 1
    assert gens == expected_gens
    assert opt.ask().shape == (0, 3)


def test_pso_and_bayes_beat_random():
    def best(name, **kw):
        opt = OPTIMIZERS.create(name, dims=4, seed=5, **kw)
        while not opt.done:
            X = opt.ask(); opt.tell(X, [_sphere(x) for x in X])
        return opt.best[1]
    n = 60
    rnd = best("random", n_samples=n)
    assert best("pso", n_particles=10, n_iterations=6) < rnd
    assert best("bayes", n_initial=20, n_iterations=8, batch_size=5, n_candidates=1000, n_restarts=1) < rnd


def test_optimizer_determinism():
    def run():
        o = OPTIMIZERS.create("pso", dims=3, seed=9, n_particles=6, n_iterations=3)
        out = []
        while not o.done:
            X = o.ask(); out.append(X); o.tell(X, [_sphere(x) for x in X])
        return np.vstack(out)
    np.testing.assert_array_equal(run(), run())
