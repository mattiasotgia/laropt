"""FHiCL template handling.

Any FHiCL structure can be tuned: laropt never edits the template in place, it
appends ``dotted.path: value`` overrides at the end (standard FHiCL practice, and
also what lets you override values pulled in through ``#include``).
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any, Iterable, List, Sequence, Set, Tuple

from .params import fcl_literal

HEADER = (
    "\n# ---------------------------------------------------------------\n"
    "# laropt overrides{tag}\n"
    "# ---------------------------------------------------------------\n"
)


class FclError(RuntimeError):
    pass


class FclTemplate:
    def __init__(self, path):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"FHiCL template not found: {self.path}")
        self.text = self.path.read_text()

    @staticmethod
    def render_text(text: str, assignments: Iterable[Tuple[str, Any]], tag: str = "") -> str:
        lines = [text.rstrip("\n"), HEADER.format(tag=f" (tag: {tag})" if tag else "")]
        for path, value in assignments:
            lines.append(f"{path}: {fcl_literal(value)}")
        return "\n".join(lines) + "\n"

    def render(self, assignments: Iterable[Tuple[str, Any]], tag: str = "") -> str:
        return self.render_text(self.text, assignments, tag)

    def write(self, assignments, out_path, tag: str = "") -> Path:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.render(assignments, tag))
        return out


# ---- optional validation through fhicl-dump (best effort) -------------------------

def fhicl_dump(path, executable: str = "fhicl-dump") -> str:
    try:
        proc = subprocess.run([executable, str(path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except FileNotFoundError as exc:
        raise FclError(f"'{executable}' not found (set up your LArSoft environment)") from exc
    if proc.returncode != 0:
        raise FclError(f"{executable} failed:\n{proc.stderr[-1500:]}")
    return proc.stdout


def flatten_dump(text: str) -> Set[str]:
    """Collect every dotted key path from ``fhicl-dump`` output (nested ``{}`` / ``[]`` format)."""
    keys: Set[str] = set()
    stack: List[str] = []
    in_seq = False
    for raw in text.splitlines():
        line = raw.split("#")[0].strip()
        if not line:
            continue
        if in_seq:
            if "]" in line:
                in_seq = False
            continue
        if line.startswith("}"):
            if stack:
                stack.pop()
            continue
        m = re.match(r"^([^\s:]+)\s*:\s*(.*)$", line)
        if not m:
            continue
        key, rest = m.groups()
        keys.add(".".join(stack + [key]))
        if rest == "{":
            stack.append(key)
        elif rest.startswith("[") and "]" not in rest:
            in_seq = True
    return keys


def check_assignments(assignments: Sequence[Tuple[str, Any]], known: Set[str]) -> List[str]:
    warnings = []
    for path, _ in assignments:
        base = re.sub(r"\[\d+\]$", "", path)
        parent = base.rsplit(".", 1)[0] if "." in base else ""
        if parent and parent not in known:
            warnings.append(f"parent table '{parent}' of '{path}' not found in the template")
        elif base not in known:
            warnings.append(f"'{path}' is not defined in the template (it will be created, not overridden)")
    return warnings
