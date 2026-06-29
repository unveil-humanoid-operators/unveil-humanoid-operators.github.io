"""pyskl.models.gcns.utils stub — loads from DS-GCN source."""

import sys
import os
import importlib.util as _ilu

_DSGCN_UTILS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.dirname(__file__))))),
    "DS-GCN", "pyskl", "models", "gcns", "utils",
)


# Symbols removed from stdlib in Python 3.12+ that some DS-GCN files import
# but never use. We patch them before loading any module that references them.
_DEAD_IMPORT_FIXES = {
    # locale.ABDAY_1 was removed in Python 3.12
    "locale": {"ABDAY_1": 131072},
    # ssl.ALERT_DESCRIPTION_CERTIFICATE_REVOKED not always available
    "ssl": {"ALERT_DESCRIPTION_CERTIFICATE_REVOKED": 42},
}

def _patch_stdlib_for_compat():
    """Add back removed symbols to already-imported stdlib modules."""
    for mod_name, attrs in _DEAD_IMPORT_FIXES.items():
        mod = sys.modules.get(mod_name)
        if mod is None:
            import importlib
            mod = importlib.import_module(mod_name)
        for attr, val in attrs.items():
            if not hasattr(mod, attr):
                setattr(mod, attr, val)

_patch_stdlib_for_compat()


def _load(name, filename):
    full_name = f"pyskl.models.gcns.utils.{name}"
    if full_name in sys.modules:
        return sys.modules[full_name]
    filepath = os.path.join(_DSGCN_UTILS, filename)
    spec = _ilu.spec_from_file_location(full_name, filepath,
                                        submodule_search_locations=[])
    mod = _ilu.module_from_spec(spec)
    mod.__package__ = "pyskl.models.gcns.utils"
    sys.modules[full_name] = mod
    spec.loader.exec_module(mod)
    return mod


# Load each util module from DS-GCN — order matters: dependencies first
_init   = _load("init_func",   "init_func.py")
_gcn    = _load("gcn",         "gcn.py")
_tcn    = _load("tcn",         "tcn.py")
_msg3d  = _load("msg3d_utils", "msg3d_utils.py")

# Expose all symbols needed by dgstgcn.py's `from .utils import ...`
unit_aagcn   = _gcn.unit_aagcn
unit_ctrgcn  = _gcn.unit_ctrgcn
unit_gcn     = _gcn.unit_gcn
dggcn        = _gcn.dggcn
dghgcn       = _gcn.dghgcn
dgphgcn      = _gcn.dgphgcn
dgphgcn1     = _gcn.dgphgcn1

unit_tcn     = _tcn.unit_tcn
mstcn        = _tcn.mstcn
dgmstcn      = _tcn.dgmstcn
dgmsmlp      = _tcn.dgmsmlp
