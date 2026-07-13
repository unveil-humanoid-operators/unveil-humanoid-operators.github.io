"""pyskl.models.gcns stub — loads DGSTGCN from DS-GCN source."""

import sys
import os
import importlib.util as _ilu

_DSGCN_GCNS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
    "DS-GCN", "pyskl", "models", "gcns",
)

# Pre-register stub for pyskl.models.gcns.utils so that
# relative imports inside dgstgcn.py resolve to the real DS-GCN utils.
from . import utils as _utils_mod  # noqa: F401  (triggers utils/__init__.py)


def _load_dsgcn_module(name, filename, extra_globals=None):
    """Load a module from DS-GCN's gcns directory under our stub package."""
    full_name = f"pyskl.models.gcns.{name}"
    if full_name in sys.modules:
        return sys.modules[full_name]

    filepath = os.path.join(_DSGCN_GCNS, filename)
    spec = _ilu.spec_from_file_location(full_name, filepath,
                                        submodule_search_locations=[])
    mod = _ilu.module_from_spec(spec)
    mod.__package__ = "pyskl.models.gcns"
    sys.modules[full_name] = mod
    if extra_globals:
        for k, v in extra_globals.items():
            setattr(mod, k, v)
    spec.loader.exec_module(mod)
    return mod


# Load dgstgcn from the real DS-GCN source
_dgstgcn_mod = _load_dsgcn_module("dgstgcn", "dgstgcn.py")
DGSTGCN = _dgstgcn_mod.DGSTGCN
