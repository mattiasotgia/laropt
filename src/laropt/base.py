"""Small shared base class for every registered component."""
from __future__ import annotations

from pathlib import Path


class Component:
    """Base for registered components; gives them a ``base_dir`` to resolve relative paths."""

    def __init__(self, base_dir=None):
        self.base_dir = Path(base_dir).resolve() if base_dir else Path.cwd()

    def resolve(self, path) -> Path:
        p = Path(str(path)).expanduser()
        return p if p.is_absolute() else (self.base_dir / p).resolve()
