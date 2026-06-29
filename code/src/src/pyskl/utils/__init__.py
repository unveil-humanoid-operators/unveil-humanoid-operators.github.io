"""pyskl.utils stub — provides Graph and cache_checkpoint."""

import sys
import os
import logging

# Load the real Graph class from DS-GCN
_DSGCN_UTILS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "DS-GCN", "pyskl", "utils",
)

# Register the real pyskl.utils submodules under our stub package
# so that `from ...utils import Graph` in dgstgcn.py resolves correctly.
import importlib.util as _ilu
import types as _types


def _load_real(name, filepath):
    """Load a module from the DS-GCN source tree and register it."""
    full_name = f"pyskl.utils._{name}"
    if full_name in sys.modules:
        return sys.modules[full_name]
    spec = _ilu.spec_from_file_location(full_name, filepath)
    mod = _ilu.module_from_spec(spec)
    sys.modules[full_name] = mod
    spec.loader.exec_module(mod)
    return mod


# Directly load the graph module (no extra dependencies)
_graph_path = os.path.join(_DSGCN_UTILS, "graph.py")
_graph_mod = _load_real("graph", _graph_path)
Graph = _graph_mod.Graph

# Also register as pyskl.utils.graph so relative imports like
# `from ....utils.graph import k_adjacency` resolve correctly
import types as _types
_graph_submod = _types.ModuleType("pyskl.utils.graph")
_graph_submod.__package__ = "pyskl.utils"
for _attr in dir(_graph_mod):
    setattr(_graph_submod, _attr, getattr(_graph_mod, _attr))
sys.modules["pyskl.utils.graph"] = _graph_submod


def cache_checkpoint(filename):
    return filename


def get_root_logger(log_file=None, log_level=logging.INFO):
    return logging.getLogger("pyskl")
