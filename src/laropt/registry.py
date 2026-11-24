"""Component registries.

Everything pluggable in laropt (optimizers, executors, metrics, objectives,
result writers, plots) lives in a :class:`Registry`, so that a YAML file can refer
to components by name and notebooks can use the very same names.

Third-party components can be added by

* decorating a class with ``@register("optimizer", "my-name")`` in any module listed
  under ``plugins:`` in the YAML (a module name or a path to a ``.py`` file), or
* declaring a setuptools entry point in the group ``laropt.<kind>s``
  (e.g. ``laropt.optimizers``).
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import sys
import warnings
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Dict, List


class Registry:
    def __init__(self, kind: str):
        self.kind = kind
        self._items: Dict[str, Any] = {}
        self._eps_loaded = False

    def register(self, name: str = None):
        def deco(obj):
            key = name or getattr(obj, "name", None) or obj.__name__.lower()
            self._items[key] = obj  # overwrite allowed: re-running a notebook cell must work
            return obj
        return deco

    def _load_entry_points(self):
        if self._eps_loaded:
            return
        self._eps_loaded = True
        group = f"laropt.{self.kind}s"
        try:
            eps = metadata.entry_points()
            selected = eps.select(group=group) if hasattr(eps, "select") else eps.get(group, [])
        except Exception:  # pragma: no cover
            return
        for ep in selected:
            try:
                self._items.setdefault(ep.name, ep.load())
            except Exception as exc:  # pragma: no cover
                warnings.warn(f"could not load {self.kind} entry point {ep.name}: {exc}")

    def get(self, name: str):
        if name not in self._items:
            self._load_entry_points()
        if name not in self._items:
            raise KeyError(f"unknown {self.kind} '{name}'. Available: {', '.join(self.names()) or '(none)'}")
        return self._items[name]

    def create(self, name: str, *args, **kwargs):
        return self.get(name)(*args, **kwargs)

    def names(self) -> List[str]:
        self._load_entry_points()
        return sorted(self._items)

    def __contains__(self, name):
        return name in self.names()

    def __repr__(self):
        return f"Registry({self.kind}: {', '.join(self.names())})"


OPTIMIZERS = Registry("optimizer")
EXECUTORS = Registry("executor")
METRICS = Registry("metric")
OBJECTIVES = Registry("objective")
WRITERS = Registry("writer")
PLOTS = Registry("plot")

_REGISTRIES = {r.kind: r for r in (OPTIMIZERS, EXECUTORS, METRICS, OBJECTIVES, WRITERS, PLOTS)}


def registry_for(kind: str) -> Registry:
    if kind not in _REGISTRIES:
        raise KeyError(f"unknown component kind '{kind}'. Kinds: {', '.join(_REGISTRIES)}")
    return _REGISTRIES[kind]


def register(kind: str, name: str = None):
    """``@register("optimizer", "my-pso")`` — add a component to the matching registry."""
    return registry_for(kind).register(name)


def _import_target(spec: str, base_dir=None):
    """Import a module by dotted name or from a ``.py`` file path."""
    p = Path(spec).expanduser()
    if not p.is_absolute() and base_dir is not None:
        p = Path(base_dir) / p
    if str(spec).endswith(".py") or p.is_file():
        p = p.resolve()
        name = f"laropt_plugin_{p.stem}_{hashlib.md5(str(p).encode()).hexdigest()[:6]}"
        if name in sys.modules:
            return sys.modules[name]
        mod_spec = importlib.util.spec_from_file_location(name, p)
        mod = importlib.util.module_from_spec(mod_spec)
        sys.modules[name] = mod
        mod_spec.loader.exec_module(mod)
        return mod
    return importlib.import_module(spec)


def load_plugins(specs, base_dir=None):
    for spec in specs or []:
        _import_target(str(spec), base_dir)


def load_callable(spec, base_dir=None) -> Callable:
    """Resolve ``'package.module:function'`` or ``'/path/file.py:function'`` (or pass a callable through)."""
    if callable(spec):
        return spec
    mod, _, fn = str(spec).rpartition(":")
    if not mod:
        raise ValueError(f"expected 'module:function', got '{spec}'")
    return getattr(_import_target(mod, base_dir), fn)
