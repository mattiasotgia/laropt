import os
import stat
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
# worker subprocesses must find laropt and the test plugin
os.environ["PYTHONPATH"] = os.pathsep.join([str(HERE.parent / "src"), str(HERE), os.environ.get("PYTHONPATH", "")])

FAKE_LAR = f"""#!{sys.executable}
import sys, argparse
sys.path.insert(0, {str(HERE)!r})
from synthetic_plugin import loss_from_fcl
ap = argparse.ArgumentParser()
ap.add_argument("-c"); ap.add_argument("-s"); ap.add_argument("--source-list")
ap.add_argument("-o"); ap.add_argument("-T"); ap.add_argument("-n")
a = ap.parse_args()
if a.T:   # analyzer run: read the 'art output' of the tuned job, emit a CSV table
    loss = float(open(a.s).read())
    open(a.T, "w").write("event,value\\n" + "\\n".join(f"{{i}},{{loss}}" for i in range(5)) + "\\n")
else:     # tuned job: its 'output' holds the loss computed from the FHiCL overrides
    text = open(a.c).read()
    if "FAIL_ME" in text: sys.exit(3)
    open(a.o, "w").write(str(loss_from_fcl(text)))
"""

TEMPLATE = "process_name: test\nphysics: {\n  hf: { Thr: 1.0  N: 2  vec: [1, 2, 3] }\n}\n"

ANALYZER = "process_name: ana\n"


@pytest.fixture
def env(tmp_path):
    lar = tmp_path / "lar"
    lar.write_text(FAKE_LAR)
    lar.chmod(lar.stat().st_mode | stat.S_IEXEC)
    (tmp_path / "template.fcl").write_text(TEMPLATE)
    (tmp_path / "ana.fcl").write_text(ANALYZER)
    (tmp_path / "in1.root").write_text("x")
    return tmp_path


@pytest.fixture
def make_config(env):
    def _make(**over):
        cfg = {
            "study": {"name": "t", "workdir": "out", "seed": 1},
            "template": "template.fcl",
            "input": {"files": "in1.root"},
            "lar": {"executable": str(env / "lar")},
            "parameters": {
                "thr": {"bounds": [0, 10], "paths": ["physics.hf.Thr"]},
                "n": {"type": "int", "bounds": [1, 6], "paths": ["physics.hf.N"]},
                "w": {"bounds": [0, 1], "size": 3, "per_element": True, "paths": ["physics.hf.vec[{i}]"]},
            },
            "metric": {"name": "synthetic"},
            "objective": {"name": "expression", "expr": "df.value.mean()"},
            "summaries": {"vmax": "df.value.max()"},
            "optimizer": {"name": "lhs", "n_samples": 12},
            "executor": {"name": "local", "workers": 1, "in_process": True},
            "writer": {"name": "sqlite"},
            "plugins": [str(HERE / "synthetic_plugin.py")],
        }
        for k, v in over.items():
            cfg[k] = v
        return cfg
    return _make
