"""
Image Feature Extraction Backbone for Explicit Constraint Branch.

Uses a lightweight CNN (e.g., ResNet-18/34) to extract multi-scale image features
from camera images. The features are then used for cross-modal fusion with
3D point cloud features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmdet3d.registry import MODELS
from mmcv.cnn import build_norm_layer
from torch import Tensor
from typing import Optional, Sequence


@MODELS.register_module()
class ImageBackbone(BaseModule):
    """Lightweight image backbone for extracting 2D features.

    Uses a simple CNN encoder to extract multi-scale features from camera images.
    Returns a feature map at a specified output stride.

    Args:
        in_channels (int): Number of input channels (3 for RGB). Default: 3.
        base_channels (int): Base number of channels. Default: 64.
        num_stages (int): Number of encoding stages. Default: 4.
        out_channels (int): Output feature channels. Default: 128.
        out_stride (int): Output feature map stride relative to input. Default: 4.
        norm_cfg (dict): Normalization config. Default: dict(type='SyncBN').
        pretrained (str, optional): Path to pretrained weights.
    """

    def __init__(self,
                 in_channels: int = 3,
                 base_channels: int = 64,
                 num_stages: int = 4,
                 out_channels: int = 128,
                 out_stride: int = 4,
                 norm_cfg: dict = dict(type='SyncBN', eps=1e-3, momentum=0.01),
                 pretrained: Optional[str] = None,
                 init_cfg: Optional[dict] = None):
        super().__init__(init_cfg=init_cfg)

        self.out_stride = out_stride

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels // 2, 3, stride=2, padding=1, bias=False),
            build_norm_layer(norm_cfg, base_channels // 2)[1],
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels // 2, base_channels, 3, stride=1, padding=1, bias=False),
            build_norm_layer(norm_cfg, base_channels)[1],
            nn.ReLU(inplace=True),
        )
        # After stem: stride=2

        # Encoder stages
        channels = [base_channels]
        self.stages = nn.ModuleList()
        in_ch = base_channels
        for i in range(num_stages):
            out_ch = base_channels * (2 ** min(i, 2))  # 64, 128, 256, 256
            stride = 2 if i > 0 else 1
            stage = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
                build_norm_layer(norm_cfg, out_ch)[1],
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False),
                build_norm_layer(norm_cfg, out_ch)[1],
                nn.ReLU(inplace=True),
            )
            self.stages.append(stage)
            channels.append(out_ch)
            in_ch = out_ch

        # Output projection: combine multi-scale features
        # Depending on out_stride, we pick the right level
        # stem: stride 2, stage0: stride 2, stage1: stride 4, stage2: stride 8, stage3: stride 16
        # So for out_stride=4: use output after stage1 (which is at stride 4 from input)
        self.out_proj = nn.Sequential(
            nn.Conv2d(in_ch, out_channels, 1, bias=False),
            build_norm_layer(norm_cfg, out_channels)[1],
            nn.ReLU(inplace=True)
        )

        self._out_channels = out_channels

        if pretrained:
            self.init_weights_from_pretrained(pretrained)

    @property
    def output_channels(self):
        return self._out_channels

    def init_weights_from_pretrained(self, pretrained: str):
        """Load pretrained weights if available."""
        import warnings
        try:
            state_dict = torch.load(pretrained, map_location='cpu')
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            missing, unexpected = self.load_state_dict(state_dict, strict=False)
            if missing:
                warnings.warn(f'Missing keys in pretrained: {missing}')
        except Exception as e:
            warnings.warn(f'Failed to load pretrained weights: {e}')

    def forward(self, images: Tensor) -> Tensor:
        """
        Args:
            images: (B, C, H, W) input camera images

        Returns:
            feat_map: (B, out_channels, H', W') feature map at out_stride
        """
        x = self.stem(x=images)  # stride 2
        for stage in self.stages:
            x = stage(x)
        x = self.out_proj(x)
        return x


@MODELS.register_module()
class ResNetImageBackbone(BaseModule):
    """ResNet-based image backbone using torchvision.

    A practical image backbone that uses a pretrained ResNet to extract features.

    Args:
        depth (int): ResNet depth (18, 34, 50). Default: 34.
        out_channels (int): Output feature channels. Default: 128.
        out_stride (int): Output stride. Features will be at this resolution.
            Default: 8 (use layer3 output).
        frozen_stages (int): Number of stages to freeze. -1 means no freezing.
        norm_cfg (dict): Normalization config.
        pretrained (bool): Whether to use ImageNet pretrained weights.
    """

    def __init__(self,
                 depth: int = 34,
                 out_channels: int = 128,
                 out_stride: int = 8,
                 frozen_stages: int = 1,
                 norm_cfg: dict = dict(type='SyncBN', eps=1e-3, momentum=0.01),
                 pretrained: bool = True,
                 init_cfg: Optional[dict] = None):
        super().__init__(init_cfg=init_cfg)

        import torchvision.models as models

        self.out_stride = out_stride
        self._out_channels = out_channels

        # Build ResNet
        if depth == 18:
            resnet = models.resnet18(pretrained=pretrained)
            feat_channels = [64, 64, 128, 256, 512]
        elif depth == 34:
            resnet = models.resnet34(pretrained=pretrained)
            feat_channels = [64, 64, 128, 256, 512]
        elif depth == 50:
            resnet = models.resnet50(pretrained=pretrained)
            feat_channels = [64, 256, 512, 1024, 2048]
        else:
            raise ValueError(f'Unsupported ResNet depth: {depth}')

        # Extract layers
        self.stem = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)  # stride 4
        self.layer1 = resnet.layer1  # stride 4
        self.layer2 = resnet.layer2  # stride 8
        self.layer3 = resnet.layer3  # stride 16
        self.layer4 = resnet.layer4  # stride 32

        # Determine which layer to use based on out_stride
        if out_stride == 4:
            feat_ch = feat_channels[1]  # layer1 output
        elif out_stride == 8:
            feat_ch = feat_channels[2]  # layer2 output
        elif out_stride == 16:
            feat_ch = feat_channels[3]  # layer3 output
        else:
            feat_ch = feat_channels[2]  # default to layer2

        # Output projection
        self.out_proj = nn.Sequential(
            nn.Conv2d(feat_ch, out_channels, 1, bias=False),
            build_norm_layer(norm_cfg, out_channels)[1],
            nn.ReLU(inplace=True)
        )

        # Freeze stages
        self.frozen_stages = frozen_stages
        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for param in self.stem.parameters():
                param.requires_grad = False
        if self.frozen_stages >= 1:
            for param in self.layer1.parameters():
                param.requires_grad = False
        if self.frozen_stages >= 2:
            for param in self.layer2.parameters():
                param.requires_grad = False
        if self.frozen_stages >= 3:
            for param in self.layer3.parameters():
                param.requires_grad = False

    @property
    def output_channels(self):
        return self._out_channels

    def forward(self, images: Tensor) -> Tensor:
        """
        Args:
            images: (B, 3, H, W) input camera images

        Returns:
            feat_map: (B, out_channels, H/out_stride, W/out_stride)
        """
        x = self.stem(images)   # stride 4
        x = self.layer1(x)     # stride 4

        if self.out_stride == 4:
            return self.out_proj(x)

        x = self.layer2(x)     # stride 8
        if self.out_stride == 8:
            return self.out_proj(x)

        x = self.layer3(x)     # stride 16
        if self.out_stride == 16:
            return self.out_proj(x)

        x = self.layer4(x)     # stride 32
        return self.out_proj(x)
