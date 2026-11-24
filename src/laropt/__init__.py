"""laropt — tune / optimise any FHiCL configuration of a LArSoft job against a user-defined metric."""
from .registry import (EXECUTORS, METRICS, OBJECTIVES, OPTIMIZERS, PLOTS, WRITERS, Registry, load_callable,
                       load_plugins, register, registry_for)
from .params import Param, ParamSpace, fcl_literal
from .fcl import FclTemplate
from .results import ResultsView
from .reduce import reduce_frame

# importing the built-ins populates the registries
from . import executors, metrics, objectives, optimizers, plotting, writers  # noqa: F401,E402
from .study import Point, Study  # noqa: E402

__version__ = "0.1.0"
__all__ = ["Study", "Point", "Param", "ParamSpace", "FclTemplate", "ResultsView", "Registry", "register",
           "registry_for", "OPTIMIZERS", "EXECUTORS", "METRICS", "OBJECTIVES", "WRITERS", "PLOTS",
           "fcl_literal", "reduce_frame", "load_callable", "load_plugins"]
