'''
Build the EfficientViT model family
'''
import torch
import torch.nn as nn
from .efficientvit import EfficientViT
from timm.models import register_model

_checkpoint_url_format = 'https://github.com/xinyuliu-jeffrey/EfficientViT_Model_Zoo/releases/download/v1.0/{}.pth'


def Replace_BatchNorm(net):
    for child_name, child in net.named_children():
        if hasattr(child, 'fuse'):
            setattr(net, child_name, child.fuse())
        elif isinstance(child, nn.BatchNorm2d):
            setattr(net, child_name, nn.Identity())
        else:
            Replace_BatchNorm(child)


EfficientViT_m0 = {
    'img_size': 224,
    'patch_size': 16,
    'embed_dim': [64, 128, 192],
    'depth': [1, 2, 3],
    'num_heads': [4, 4, 4],
    'window_size': [7, 7, 7],
    'kernels': [5, 5, 5, 5],
}

EfficientViT_m1 = {
    'img_size': 224,
    'patch_size': 16,
    'embed_dim': [128, 144, 192],
    'depth': [1, 2, 3],
    'num_heads': [2, 3, 3],
    'window_size': [7, 7, 7],
    'kernels': [7, 5, 3, 3],
}

EfficientViT_m2 = {
    'img_size': 224,
    'patch_size': 16,
    'embed_dim': [128, 192, 224],
    'depth': [1, 2, 3],
    'num_heads': [4, 3, 2],
    'window_size': [7, 7, 7],
    'kernels': [7, 5, 3, 3],
}

EfficientViT_m3 = {
    'img_size': 224,
    'patch_size': 16,
    'embed_dim': [128, 240, 320],
    'depth': [1, 2, 3],
    'num_heads': [4, 3, 4],
    'window_size': [7, 7, 7],
    'kernels': [5, 5, 5, 5],
}

EfficientViT_m4 = {
    'img_size': 224,
    'patch_size': 16,
    'embed_dim': [128, 256, 384],
    'depth': [1, 2, 3],
    'num_heads': [4, 4, 4],
    'window_size': [7, 7, 7],
    'kernels': [7, 5, 3, 3],
}

EfficientViT_m5 = {
    'img_size': 224,
    'patch_size': 16,
    'embed_dim': [192, 288, 384],
    'depth': [1, 3, 4],
    'num_heads': [3, 3, 4],
    'window_size': [7, 7, 7],
    'kernels': [7, 5, 3, 3],
}


@register_model
def EfficientViT_M0(num_classes=1000, pretrained=False, distillation=False, fuse=False, pretrained_cfg=None,
                    pretrained_cfg_overlay=None,
                    model_cfg=EfficientViT_m0):
    model = EfficientViT(num_classes=num_classes, distillation=distillation, **model_cfg)
    if pretrained:
        pretrained = _checkpoint_url_format.format(pretrained)
        checkpoint = torch.hub.load_state_dict_from_url(
            pretrained, map_location='cpu')
        d = checkpoint['model']
        D = model.state_dict()
        for k in d.keys():
            if D[k].shape != d[k].shape:
                d[k] = d[k][:, :, None, None]
        model.load_state_dict(d)
    if fuse:
        Replace_BatchNorm(model)
    return model


@register_model
def EfficientViT_M1(num_classes=1000, pretrained=False, distillation=False, fuse=False, pretrained_cfg=None,
                    pretrained_cfg_overlay=None,
                    model_cfg=EfficientViT_m1):
    model = EfficientViT(num_classes=num_classes, distillation=distillation, **model_cfg)
    if pretrained:
        pretrained = _checkpoint_url_format.format(pretrained)
        checkpoint = torch.hub.load_state_dict_from_url(
            pretrained, map_location='cpu')
        d = checkpoint['model']
        D = model.state_dict()
        for k in d.keys():
            if D[k].shape != d[k].shape:
                d[k] = d[k][:, :, None, None]
        model.load_state_dict(d)
    if fuse:
        Replace_BatchNorm(model)
    return model


@register_model
def EfficientViT_M2(num_classes=1000, pretrained=False, distillation=False, fuse=False, pretrained_cfg=None,
                    pretrained_cfg_overlay=None,
                    model_cfg=EfficientViT_m2):
    model = EfficientViT(num_classes=num_classes, distillation=distillation, **model_cfg)
    if pretrained:
        pretrained = _checkpoint_url_format.format(pretrained)
        checkpoint = torch.hub.load_state_dict_from_url(
            pretrained, map_location='cpu')
        d = checkpoint['model']
        D = model.state_dict()
        for k in d.keys():
            if D[k].shape != d[k].shape:
                d[k] = d[k][:, :, None, None]
        model.load_state_dict(d)
    if fuse:
        Replace_BatchNorm(model)
    return model


@register_model
def EfficientViT_M3(num_classes=1000, pretrained=False, distillation=False, fuse=False, pretrained_cfg=None,
                    pretrained_cfg_overlay=None,
                    model_cfg=EfficientViT_m3):
    model = EfficientViT(num_classes=num_classes, distillation=distillation, **model_cfg)
    if pretrained:
        pretrained = _checkpoint_url_format.format(pretrained)
        checkpoint = torch.hub.load_state_dict_from_url(
            pretrained, map_location='cpu')
        d = checkpoint['model']
        D = model.state_dict()
        for k in d.keys():
            if D[k].shape != d[k].shape:
                d[k] = d[k][:, :, None, None]
        model.load_state_dict(d)
    if fuse:
        Replace_BatchNorm(model)
    return model


@register_model
def EfficientViT_M4(num_classes=1000, pretrained=False, distillation=False, fuse=False, pretrained_cfg=None,
                    pretrained_cfg_overlay=None,
                    model_cfg=EfficientViT_m4):
    model = EfficientViT(num_classes=num_classes, distillation=distillation, **model_cfg)
    if pretrained:
        pretrained = _checkpoint_url_format.format(pretrained)
        checkpoint = torch.hub.load_state_dict_from_url(
            pretrained, map_location='cpu')
        d = checkpoint['model']
        D = model.state_dict()
        for k in d.keys():
            if D[k].shape != d[k].shape:
                d[k] = d[k][:, :, None, None]
        model.load_state_dict(d)
    if fuse:
        Replace_BatchNorm(model)
    return model


@register_model
def EfficientViT_M5(num_classes=1000, pretrained=False, distillation=False, fuse=False, pretrained_cfg=None,
                    pretrained_cfg_overlay=None,
                    model_cfg=EfficientViT_m5):
    model = EfficientViT(num_classes=num_classes, distillation=distillation, **model_cfg)
    if pretrained:
        pretrained = _checkpoint_url_format.format(pretrained)
        checkpoint = torch.hub.load_state_dict_from_url(
            pretrained, map_location='cpu')
        d = checkpoint['model']
        D = model.state_dict()
        for k in d.keys():
            if D[k].shape != d[k].shape:
                d[k] = d[k][:, :, None, None]
        model.load_state_dict(d)
    if fuse:
        Replace_BatchNorm(model)
    return model
