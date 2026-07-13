"""Minimal pyskl stub — only exposes DGSTGCN for inference.

This shadows the full DS-GCN/pyskl package so we can import DGSTGCN
without pulling in decord, mmcv training runners, video datasets, etc.
All model forward-pass logic is loaded directly from DS-GCN/pyskl/models/gcns/.
"""

import sys
import os

# Point at the real DS-GCN source for actual model code
_DSGCN_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "DS-GCN")

# We will lazily populate submodule stubs when needed.
# The key thing is that 'import pyskl' no longer chains through
# datasets/apis/decord/cv2 etc.
