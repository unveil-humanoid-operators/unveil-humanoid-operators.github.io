"""Minimal model registry (trimmed from the upstream builder)."""

from mmcv.utils import Registry

MODELS = Registry('models')
BACKBONES = MODELS


def build_backbone(cfg):
    """Build backbone."""
    return BACKBONES.build(cfg)
