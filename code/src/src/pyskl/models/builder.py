"""pyskl.models.builder stub."""
from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.utils import Registry

MODELS = Registry("models", parent=MMCV_MODELS)
BACKBONES = MODELS
NECKS = MODELS
HEADS = MODELS
RECOGNIZERS = MODELS
LOSSES = MODELS


def build_backbone(cfg):
    return BACKBONES.build(cfg)


def build_head(cfg):
    return HEADS.build(cfg)


def build_loss(cfg):
    return LOSSES.build(cfg)


def build_model(cfg, train_cfg=None, test_cfg=None):
    return MODELS.build(cfg)
