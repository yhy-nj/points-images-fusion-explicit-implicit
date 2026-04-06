# """
# Modified FRNet Backbone with Explicit Constraint Branch.
#
# This backbone extends the original FRNetBackbone by adding an explicit
# constraint branch that fuses image features with 3D point cloud features.
# The explicit constraint is applied at the output of the backbone (after
# feature fusion head), operating on the final fused point features.
#
# Architecture:
# 1. Original FRNet backbone processes point cloud as before
# 2. Image backbone extracts features from camera images
# 3. After FRNet backbone output, the explicit constraint branch:
#    a. Projects 3D points to 2D image plane
#    b. Predicts offsets and weights for deformable feature correction
#    c. Performs gated cross-modal fusion (2DPASS-style)
# 4. Enhanced features are concatenated/fused with original backbone output
# """
#
# from typing import Optional, Sequence, Tuple
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch_scatter
# from mmcv.cnn import (ConvModule, build_activation_layer, build_conv_layer,
#                       build_norm_layer)
# from mmdet3d.registry import MODELS
# from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig
# from mmengine.model import BaseModule
# from torch import Tensor
#
# from .explicit_fusion import ExplicitConstraintBranch
#
#
# class BasicBlock(BaseModule):
#     """Same BasicBlock as original FRNet."""
#
#     def __init__(self,
#                  inplanes: int,
#                  planes: int,
#                  stride: int = 1,
#                  dilation: int = 1,
#                  downsample: Optional[nn.Module] = None,
#                  conv_cfg: OptConfigType = None,
#                  norm_cfg: ConfigType = dict(type='BN'),
#                  act_cfg: ConfigType = dict(type='LeakyReLU'),
#                  init_cfg: OptMultiConfig = None) -> None:
#         super(BasicBlock, self).__init__(init_cfg)
#
#         self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
#         self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)
#
#         self.conv1 = build_conv_layer(
#             conv_cfg, inplanes, planes, 3,
#             stride=stride, padding=dilation, dilation=dilation, bias=False)
#         self.add_module(self.norm1_name, norm1)
#         self.conv2 = build_conv_layer(
#             conv_cfg, planes, planes, 3, padding=1, bias=False)
#         self.add_module(self.norm2_name, norm2)
#         self.relu = build_activation_layer(act_cfg)
#         self.downsample = downsample
#
#     @property
#     def norm1(self):
#         return getattr(self, self.norm1_name)
#
#     @property
#     def norm2(self):
#         return getattr(self, self.norm2_name)
#
#     def forward(self, x: Tensor) -> Tensor:
#         identity = x
#         out = self.conv1(x)
#         out = self.norm1(out)
#         out = self.relu(out)
#         out = self.conv2(out)
#         out = self.norm2(out)
#         if self.downsample is not None:
#             identity = self.downsample(x)
#         out += identity
#         out = self.relu(out)
#         return out
#
#
# @MODELS.register_module()
# class FRNetExplicitBackbone(BaseModule):
#     """FRNet Backbone with Explicit Constraint Branch.
#
#     This extends FRNetBackbone by adding image-based explicit constraint
#     fusion at the backbone output level.
#
#     Additional Args (compared to FRNetBackbone):
#         image_backbone_cfg (dict): Config for the image backbone.
#         explicit_voxel_channels (int): Voxel feature channels for explicit branch.
#         explicit_image_channels (int): Image feature channels for explicit branch.
#         explicit_align_channels (int): Aligned feature channels.
#         explicit_out_channels (int): Output channels of explicit branch.
#         explicit_num_samples (int): Number of deformable sampling points.
#         enable_explicit (bool): Whether to enable the explicit constraint.
#     """
#
#     arch_settings = {
#         18: (BasicBlock, (2, 2, 2, 2)),
#         34: (BasicBlock, (3, 4, 6, 3))
#     }
#
#     def __init__(self,
#                  in_channels: int,
#                  point_in_channels: int,
#                  output_shape: Sequence[int],
#                  depth: int,
#                  stem_channels: int = 128,
#                  num_stages: int = 4,
#                  out_channels: Sequence[int] = (128, 128, 128, 128),
#                  strides: Sequence[int] = (1, 2, 2, 2),
#                  dilations: Sequence[int] = (1, 1, 1, 1),
#                  fuse_channels: Sequence[int] = (256, 128),
#                  conv_cfg: OptConfigType = None,
#                  norm_cfg: ConfigType = dict(type='BN'),
#                  point_norm_cfg: ConfigType = dict(type='BN1d'),
#                  act_cfg: ConfigType = dict(type='LeakyReLU'),
#                  # Explicit constraint parameters
#                  image_backbone_cfg: Optional[dict] = None,
#                  explicit_voxel_channels: int = 128,
#                  explicit_image_channels: int = 128,
#                  explicit_align_channels: int = 128,
#                  explicit_out_channels: int = 128,
#                  explicit_num_samples: int = 9,
#                  enable_explicit: bool = True,
#                  init_cfg: OptMultiConfig = None) -> None:
#         super().__init__(init_cfg)
#
#         if depth not in self.arch_settings:
#             raise KeyError(f'invalid depth {depth} for FRNetBackbone.')
#
#         self.block, stage_blocks = self.arch_settings[depth]
#         self.output_shape = output_shape
#         self.ny = output_shape[0]
#         self.nx = output_shape[1]
#         assert len(stage_blocks) == len(out_channels) == len(strides) == len(
#             dilations) == num_stages
#
#         self.conv_cfg = conv_cfg
#         self.norm_cfg = norm_cfg
#         self.point_norm_cfg = point_norm_cfg
#         self.act_cfg = act_cfg
#         self.enable_explicit = enable_explicit
#
#         # =====================================================
#         # Original FRNet backbone components (unchanged)
#         # =====================================================
#         self.stem = self._make_stem_layer(in_channels, stem_channels)
#         self.point_stem = self._make_point_layer(point_in_channels, stem_channels)
#         self.fusion_stem = self._make_fusion_layer(stem_channels * 2, stem_channels)
#
#         inplanes = stem_channels
#         self.res_layers = []
#         self.point_fusion_layers = nn.ModuleList()
#         self.pixel_fusion_layers = nn.ModuleList()
#         self.attention_layers = nn.ModuleList()
#         self.strides = []
#         overall_stride = 1
#         for i, num_blocks in enumerate(stage_blocks):
#             stride = strides[i]
#             overall_stride = stride * overall_stride
#             self.strides.append(overall_stride)
#             dilation = dilations[i]
#             planes = out_channels[i]
#             res_layer = self.make_res_layer(
#                 block=self.block, inplanes=inplanes, planes=planes,
#                 num_blocks=num_blocks, stride=stride, dilation=dilation,
#                 conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
#             self.point_fusion_layers.append(
#                 self._make_point_layer(inplanes + planes, planes))
#             self.pixel_fusion_layers.append(
#                 self._make_fusion_layer(planes * 2, planes))
#             self.attention_layers.append(self._make_attention_layer(planes))
#             inplanes = planes
#             layer_name = f'layer{i + 1}'
#             self.add_module(layer_name, res_layer)
#             self.res_layers.append(layer_name)
#
#         in_channels_fuse = stem_channels + sum(out_channels)
#         self.fuse_layers = []
#         self.point_fuse_layers = []
#         for i, fuse_channel in enumerate(fuse_channels):
#             fuse_layer = ConvModule(
#                 in_channels_fuse, fuse_channel,
#                 kernel_size=3, padding=1,
#                 conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
#             point_fuse_layer = self._make_point_layer(
#                 in_channels_fuse, fuse_channel)
#             in_channels_fuse = fuse_channel
#             layer_name = f'fuse_layer{i + 1}'
#             point_layer_name = f'point_fuse_layer{i + 1}'
#             self.add_module(layer_name, fuse_layer)
#             self.add_module(point_layer_name, point_fuse_layer)
#             self.fuse_layers.append(layer_name)
#             self.point_fuse_layers.append(point_layer_name)
#
#         # =====================================================
#         # Explicit Constraint Branch components (NEW)
#         # =====================================================
#         if enable_explicit:
#             # Image backbone
#             if image_backbone_cfg is not None:
#                 self.image_backbone = MODELS.build(image_backbone_cfg)
#                 actual_image_channels = self.image_backbone.output_channels
#             else:
#                 # Default lightweight image backbone
#                 from .image_backbone import ImageBackbone
#                 self.image_backbone = ImageBackbone(
#                     in_channels=3,
#                     base_channels=64,
#                     num_stages=3,
#                     out_channels=explicit_image_channels,
#                     out_stride=8)
#                 actual_image_channels = explicit_image_channels
#
#             # Explicit constraint fusion module
#             # Applied after the backbone fusion head output
#             # voxel_channels = fuse_channels[-1] (final point feature dim)
#             final_point_channels = fuse_channels[-1]
#             self.explicit_branch = ExplicitConstraintBranch(
#                 voxel_channels=final_point_channels,
#                 image_channels=actual_image_channels,
#                 image_align_channels=explicit_align_channels,
#                 out_channels=explicit_out_channels,
#                 num_samples=explicit_num_samples,
#                 norm_cfg=point_norm_cfg
#             )
#
#             # Final merge: combine original backbone output with explicit branch output
#             # Original point features: fuse_channels[-1]
#             # Explicit branch output: explicit_out_channels
#             self.explicit_merge = nn.Sequential(
#                 nn.Linear(final_point_channels + explicit_out_channels,
#                           final_point_channels, bias=False),
#                 build_norm_layer(point_norm_cfg, final_point_channels)[1],
#                 nn.ReLU(inplace=True)
#             )
#
#     # =====================================================
#     # Layer construction methods (same as original FRNet)
#     # =====================================================
#     def _make_stem_layer(self, in_channels, out_channels):
#         return nn.Sequential(
#             build_conv_layer(self.conv_cfg, in_channels, out_channels // 2,
#                              kernel_size=3, padding=1, bias=False),
#             build_norm_layer(self.norm_cfg, out_channels // 2)[1],
#             build_activation_layer(self.act_cfg),
#             build_conv_layer(self.conv_cfg, out_channels // 2, out_channels,
#                              kernel_size=3, padding=1, bias=False),
#             build_norm_layer(self.norm_cfg, out_channels)[1],
#             build_activation_layer(self.act_cfg),
#             build_conv_layer(self.conv_cfg, out_channels, out_channels,
#                              kernel_size=3, padding=1, bias=False),
#             build_norm_layer(self.norm_cfg, out_channels)[1],
#             build_activation_layer(self.act_cfg))
#
#     def _make_point_layer(self, in_channels, out_channels):
#         return nn.Sequential(
#             nn.Linear(in_channels, out_channels, bias=False),
#             build_norm_layer(self.point_norm_cfg, out_channels)[1],
#             nn.ReLU(inplace=True))
#
#     def _make_fusion_layer(self, in_channels, out_channels):
#         return nn.Sequential(
#             build_conv_layer(self.conv_cfg, in_channels, out_channels,
#                              kernel_size=3, padding=1, bias=False),
#             build_norm_layer(self.norm_cfg, out_channels)[1],
#             build_activation_layer(self.act_cfg))
#
#     def _make_attention_layer(self, channels):
#         return nn.Sequential(
#             build_conv_layer(self.conv_cfg, channels, channels,
#                              kernel_size=3, padding=1, bias=False),
#             build_norm_layer(self.norm_cfg, channels)[1],
#             build_activation_layer(self.act_cfg),
#             build_conv_layer(self.conv_cfg, channels, channels,
#                              kernel_size=3, padding=1, bias=False),
#             build_norm_layer(self.norm_cfg, channels)[1],
#             nn.Sigmoid())
#
#     def make_res_layer(self, block, inplanes, planes, num_blocks,
#                        stride, dilation, conv_cfg=None,
#                        norm_cfg=dict(type='BN'),
#                        act_cfg=dict(type='LeakyReLU')):
#         downsample = None
#         if stride != 1 or inplanes != planes:
#             downsample = nn.Sequential(
#                 build_conv_layer(conv_cfg, inplanes, planes,
#                                  kernel_size=1, stride=stride, bias=False),
#                 build_norm_layer(norm_cfg, planes)[1])
#         layers = []
#         layers.append(block(
#             inplanes=inplanes, planes=planes, stride=stride,
#             dilation=dilation, downsample=downsample,
#             conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg))
#         inplanes = planes
#         for _ in range(1, num_blocks):
#             layers.append(block(
#                 inplanes=inplanes, planes=planes, stride=1,
#                 dilation=dilation, conv_cfg=conv_cfg,
#                 norm_cfg=norm_cfg, act_cfg=act_cfg))
#         return nn.Sequential(*layers)
#
#     # =====================================================
#     # Projection utility methods (same as original FRNet)
#     # =====================================================
#     def frustum2pixel(self, frustum_features, coors, batch_size, stride=1):
#         nx = self.nx // stride
#         ny = self.ny // stride
#         pixel_features = torch.zeros(
#             (batch_size, ny, nx, frustum_features.shape[-1]),
#             dtype=frustum_features.dtype, device=frustum_features.device)
#         pixel_features[coors[:, 0], coors[:, 1], coors[:, 2]] = frustum_features
#         pixel_features = pixel_features.permute(0, 3, 1, 2).contiguous()
#         return pixel_features
#
#     def pixel2point(self, pixel_features, coors, stride=1):
#         pixel_features = pixel_features.permute(0, 2, 3, 1).contiguous()
#         point_feats = pixel_features[coors[:, 0], coors[:, 1] // stride,
#                                      coors[:, 2] // stride]
#         return point_feats
#
#     def point2frustum(self, point_features, pts_coors, stride=1):
#         coors = pts_coors.clone()
#         coors[:, 1] = pts_coors[:, 1] // stride
#         coors[:, 2] = pts_coors[:, 2] // stride
#         voxel_coors, inverse_map = torch.unique(
#             coors, return_inverse=True, dim=0)
#         frustum_features = torch_scatter.scatter_max(
#             point_features, inverse_map, dim=0)[0]
#         return voxel_coors, frustum_features
#
#     # =====================================================
#     # Forward pass
#     # =====================================================
#     def forward(self, voxel_dict: dict) -> dict:
#         """Forward pass with explicit constraint branch.
#
#         The original FRNet backbone forward is preserved. After backbone
#         feature extraction, if images are available, the explicit constraint
#         branch enhances the point features using image information.
#         """
#         # ---- Original FRNet backbone forward (unchanged) ----
#         point_feats = voxel_dict['point_feats'][-1]
#         voxel_feats = voxel_dict['voxel_feats']
#         voxel_coors = voxel_dict['voxel_coors']
#         pts_coors = voxel_dict['coors']
#         batch_size = pts_coors[-1, 0].item() + 1
#
#         x = self.frustum2pixel(voxel_feats, voxel_coors, batch_size, stride=1)
#         x = self.stem(x)
#         map_point_feats = self.pixel2point(x, pts_coors, stride=1)
#         fusion_point_feats = torch.cat((map_point_feats, point_feats), dim=1)
#         point_feats = self.point_stem(fusion_point_feats)
#         stride_voxel_coors, frustum_feats = self.point2frustum(
#             point_feats, pts_coors, stride=1)
#         pixel_feats = self.frustum2pixel(
#             frustum_feats, stride_voxel_coors, batch_size, stride=1)
#         fusion_pixel_feats = torch.cat((pixel_feats, x), dim=1)
#         x = self.fusion_stem(fusion_pixel_feats)
#
#         outs = [x]
#         out_points = [point_feats]
#         for i, layer_name in enumerate(self.res_layers):
#             res_layer = getattr(self, layer_name)
#             x = res_layer(x)
#
#             # frustum-to-point fusion
#             map_point_feats = self.pixel2point(
#                 x, pts_coors, stride=self.strides[i])
#             fusion_point_feats = torch.cat(
#                 (map_point_feats, point_feats), dim=1)
#             point_feats = self.point_fusion_layers[i](fusion_point_feats)
#
#             # point-to-frustum fusion
#             stride_voxel_coors, frustum_feats = self.point2frustum(
#                 point_feats, pts_coors, stride=self.strides[i])
#             pixel_feats = self.frustum2pixel(
#                 frustum_feats, stride_voxel_coors, batch_size,
#                 stride=self.strides[i])
#             fusion_pixel_feats = torch.cat((pixel_feats, x), dim=1)
#             fuse_out = self.pixel_fusion_layers[i](fusion_pixel_feats)
#             # residual-attentive
#             attention_map = self.attention_layers[i](fuse_out)
#             x = fuse_out * attention_map + x
#             outs.append(x)
#             out_points.append(point_feats)
#
#         for i in range(len(outs)):
#             if outs[i].shape != outs[0].shape:
#                 outs[i] = F.interpolate(
#                     outs[i], size=outs[0].size()[2:],
#                     mode='bilinear', align_corners=True)
#
#         outs[0] = torch.cat(outs, dim=1)
#         out_points[0] = torch.cat(out_points, dim=1)
#
#         for layer_name, point_layer_name in zip(
#                 self.fuse_layers, self.point_fuse_layers):
#             fuse_layer = getattr(self, layer_name)
#             outs[0] = fuse_layer(outs[0])
#             point_fuse_layer = getattr(self, point_layer_name)
#             out_points[0] = point_fuse_layer(out_points[0])
#
#         # ---- Explicit Constraint Branch (NEW) ----
#         if self.enable_explicit and voxel_dict.get('has_images', False):
#             # Extract image features
#             images = voxel_dict['images']  # (B, 3, H_img, W_img)
#             image_feat_map = self.image_backbone(images)  # (B, C_img, H', W')
#
#             # Get projection coordinates
#             proj_coords = voxel_dict['proj_coords']  # (N, 3) [batch, y, x]
#
#             # Get final point features from backbone
#             backbone_point_feats = out_points[0]  # (N, C_fuse)
#
#             # Apply explicit constraint branch
#             explicit_feats = self.explicit_branch(
#                 voxel_feats=backbone_point_feats,
#                 image_feat_map=image_feat_map,
#                 proj_coords=proj_coords
#             )  # (N, C_explicit)
#
#             # Merge with original backbone output
#             merged_point_feats = self.explicit_merge(
#                 torch.cat([backbone_point_feats, explicit_feats], dim=-1))
#             out_points[0] = merged_point_feats
#
#         # ---- Store results ----
#         voxel_dict['voxel_feats'] = outs
#         voxel_dict['point_feats_backbone'] = out_points
#         return voxel_dict

"""
Modified FRNet Backbone with Explicit Constraint Branch.

This backbone extends the original FRNetBackbone by adding an explicit
constraint branch that fuses image features with 3D point cloud features.
The explicit constraint is applied at the output of the backbone (after
feature fusion head), operating on the final fused point features.

Architecture:
1. Original FRNet backbone processes point cloud as before
2. Image backbone extracts features from camera images
3. After FRNet backbone output, the explicit constraint branch:
   a. Projects 3D points to 2D image plane
   b. Predicts offsets and weights for deformable feature correction
   c. Performs gated cross-modal fusion (2DPASS-style)
4. Enhanced features are concatenated/fused with original backbone output
"""

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from mmcv.cnn import (ConvModule, build_activation_layer, build_conv_layer,
                      build_norm_layer)
from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig
from mmengine.model import BaseModule
from open3d.cuda.pybind.geometry import Voxel
from torch import Tensor

from .explicit_fusion import ExplicitConstraintBranch


class BasicBlock(BaseModule):
    """Same BasicBlock as original FRNet."""

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: int = 1,
                 dilation: int = 1,
                 downsample: Optional[nn.Module] = None,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(type='BN'),
                 act_cfg: ConfigType = dict(type='LeakyReLU'),
                 init_cfg: OptMultiConfig = None) -> None:
        super(BasicBlock, self).__init__(init_cfg)

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)

        self.conv1 = build_conv_layer(
            conv_cfg, inplanes, planes, 3,
            stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg, planes, planes, 3, padding=1, bias=False)
        self.add_module(self.norm2_name, norm2)
        self.relu = build_activation_layer(act_cfg)
        self.downsample = downsample

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


@MODELS.register_module()
class FRNetExplicitBackbone(BaseModule):
    """FRNet Backbone with Explicit Constraint Branch.

    This extends FRNetBackbone by adding image-based explicit constraint
    fusion at the backbone output level.

    Additional Args (compared to FRNetBackbone):
        image_backbone_cfg (dict): Config for the image backbone.
        explicit_voxel_channels (int): Voxel feature channels for explicit branch.
        explicit_image_channels (int): Image feature channels for explicit branch.
        explicit_align_channels (int): Aligned feature channels.
        explicit_out_channels (int): Output channels of explicit branch.
        explicit_num_samples (int): Number of deformable sampling points.
        enable_explicit (bool): Whether to enable the explicit constraint.
    """

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3))
    }

    def __init__(self,
                 in_channels: int,
                 point_in_channels: int,
                 output_shape: Sequence[int],
                 depth: int,
                 stem_channels: int = 128,
                 num_stages: int = 4,
                 out_channels: Sequence[int] = (128, 128, 128, 128),
                 strides: Sequence[int] = (1, 2, 2, 2),
                 dilations: Sequence[int] = (1, 1, 1, 1),
                 fuse_channels: Sequence[int] = (256, 128),
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(type='BN'),
                 point_norm_cfg: ConfigType = dict(type='BN1d'),
                 act_cfg: ConfigType = dict(type='LeakyReLU'),
                 # Explicit constraint parameters
                 image_backbone_cfg: Optional[dict] = None,
                 explicit_voxel_channels: int = 128,
                 explicit_image_channels: int = 128,
                 explicit_align_channels: int = 128,
                 explicit_out_channels: int = 128,
                 explicit_num_samples: int = 9,
                 enable_explicit: bool = True,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg)

        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for FRNetBackbone.')

        self.block, stage_blocks = self.arch_settings[depth]
        self.output_shape = output_shape
        self.ny = output_shape[0]
        self.nx = output_shape[1]
        assert len(stage_blocks) == len(out_channels) == len(strides) == len(
            dilations) == num_stages

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.point_norm_cfg = point_norm_cfg
        self.act_cfg = act_cfg
        self.enable_explicit = enable_explicit

        # =====================================================
        # Original FRNet backbone components (unchanged)
        # =====================================================
        self.stem = self._make_stem_layer(in_channels, stem_channels)
        self.point_stem = self._make_point_layer(point_in_channels, stem_channels)
        self.fusion_stem = self._make_fusion_layer(stem_channels * 2, stem_channels)

        inplanes = stem_channels
        self.res_layers = []
        self.point_fusion_layers = nn.ModuleList()
        self.pixel_fusion_layers = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.strides = []
        overall_stride = 1
        for i, num_blocks in enumerate(stage_blocks):
            stride = strides[i]
            overall_stride = stride * overall_stride
            self.strides.append(overall_stride)
            dilation = dilations[i]
            planes = out_channels[i]
            res_layer = self.make_res_layer(
                block=self.block, inplanes=inplanes, planes=planes,
                num_blocks=num_blocks, stride=stride, dilation=dilation,
                conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
            self.point_fusion_layers.append(
                self._make_point_layer(inplanes + planes, planes))
            self.pixel_fusion_layers.append(
                self._make_fusion_layer(planes * 2, planes))
            self.attention_layers.append(self._make_attention_layer(planes))
            inplanes = planes
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        in_channels_fuse = stem_channels + sum(out_channels)
        self.fuse_layers = []
        self.point_fuse_layers = []
        for i, fuse_channel in enumerate(fuse_channels):
            fuse_layer = ConvModule(
                in_channels_fuse, fuse_channel,
                kernel_size=3, padding=1,
                conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
            point_fuse_layer = self._make_point_layer(
                in_channels_fuse, fuse_channel)
            in_channels_fuse = fuse_channel
            layer_name = f'fuse_layer{i + 1}'
            point_layer_name = f'point_fuse_layer{i + 1}'
            self.add_module(layer_name, fuse_layer)
            self.add_module(point_layer_name, point_fuse_layer)
            self.fuse_layers.append(layer_name)
            self.point_fuse_layers.append(point_layer_name)

        # =====================================================
        # Explicit Constraint Branch components (NEW)
        # =====================================================
        if enable_explicit:
            # Image backbone
            if image_backbone_cfg is not None:
                self.image_backbone = MODELS.build(image_backbone_cfg)
                actual_image_channels = self.image_backbone.output_channels
            else:
                # Default lightweight image backbone
                from .image_backbone import ImageBackbone
                self.image_backbone = ImageBackbone(
                    in_channels=3,
                    base_channels=64,
                    num_stages=3,
                    out_channels=explicit_image_channels,
                    out_stride=8)
                actual_image_channels = explicit_image_channels

            # Explicit constraint fusion module
            # Applied after the backbone fusion head output
            # voxel_channels = fuse_channels[-1] (final point feature dim)
            final_point_channels = fuse_channels[-1]
            self.explicit_branch = ExplicitConstraintBranch(
                voxel_channels=final_point_channels,
                image_channels=actual_image_channels,
                image_align_channels=explicit_align_channels,
                out_channels=explicit_out_channels,
                num_samples=explicit_num_samples,
                norm_cfg=point_norm_cfg
            )

            # Final merge: combine original backbone output with explicit branch output
            # Original point features: fuse_channels[-1]
            # Explicit branch output: explicit_out_channels
            self.explicit_merge = nn.Sequential(
                nn.Linear(final_point_channels + explicit_out_channels,
                          final_point_channels, bias=False),
                build_norm_layer(point_norm_cfg, final_point_channels)[1],
                nn.ReLU(inplace=True)
            )

            # ---- Contrastive Alignment Heads (φ_V and φ_I) ----
            # These project F_voxel and F'_image into a shared embedding space
            # for contrastive loss computation (only used during training)
            from frnet.models.losses.contrastive_loss import ContrastiveProjectionHead
            contrastive_embed_dim = 64
            self.proj_head_voxel = ContrastiveProjectionHead(
                in_channels=final_point_channels,
                proj_channels=final_point_channels,
                out_channels=contrastive_embed_dim,
                norm_cfg=point_norm_cfg
            )
            self.proj_head_image = ContrastiveProjectionHead(
                in_channels=explicit_align_channels,
                proj_channels=explicit_align_channels,
                out_channels=contrastive_embed_dim,
                norm_cfg=point_norm_cfg
            )

    # =====================================================
    # Layer construction methods (same as original FRNet)
    # =====================================================
    def _make_stem_layer(self, in_channels, out_channels):
        return nn.Sequential(
            build_conv_layer(self.conv_cfg, in_channels, out_channels // 2,
                             kernel_size=3, padding=1, bias=False),
            build_norm_layer(self.norm_cfg, out_channels // 2)[1],
            build_activation_layer(self.act_cfg),
            build_conv_layer(self.conv_cfg, out_channels // 2, out_channels,
                             kernel_size=3, padding=1, bias=False),
            build_norm_layer(self.norm_cfg, out_channels)[1],
            build_activation_layer(self.act_cfg),
            build_conv_layer(self.conv_cfg, out_channels, out_channels,
                             kernel_size=3, padding=1, bias=False),
            build_norm_layer(self.norm_cfg, out_channels)[1],
            build_activation_layer(self.act_cfg))

    def _make_point_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Linear(in_channels, out_channels, bias=False),
            build_norm_layer(self.point_norm_cfg, out_channels)[1],
            nn.ReLU(inplace=True))

    def _make_fusion_layer(self, in_channels, out_channels):
        return nn.Sequential(
            build_conv_layer(self.conv_cfg, in_channels, out_channels,
                             kernel_size=3, padding=1, bias=False),
            build_norm_layer(self.norm_cfg, out_channels)[1],
            build_activation_layer(self.act_cfg))

    def _make_attention_layer(self, channels):
        return nn.Sequential(
            build_conv_layer(self.conv_cfg, channels, channels,
                             kernel_size=3, padding=1, bias=False),
            build_norm_layer(self.norm_cfg, channels)[1],
            build_activation_layer(self.act_cfg),
            build_conv_layer(self.conv_cfg, channels, channels,
                             kernel_size=3, padding=1, bias=False),
            build_norm_layer(self.norm_cfg, channels)[1],
            nn.Sigmoid())

    def make_res_layer(self, block, inplanes, planes, num_blocks,
                       stride, dilation, conv_cfg=None,
                       norm_cfg=dict(type='BN'),
                       act_cfg=dict(type='LeakyReLU')):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                build_conv_layer(conv_cfg, inplanes, planes,
                                 kernel_size=1, stride=stride, bias=False),
                build_norm_layer(norm_cfg, planes)[1])
        layers = []
        layers.append(block(
            inplanes=inplanes, planes=planes, stride=stride,
            dilation=dilation, downsample=downsample,
            conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg))
        inplanes = planes
        for _ in range(1, num_blocks):
            layers.append(block(
                inplanes=inplanes, planes=planes, stride=1,
                dilation=dilation, conv_cfg=conv_cfg,
                norm_cfg=norm_cfg, act_cfg=act_cfg))
        return nn.Sequential(*layers)

    # =====================================================
    # Projection utility methods (same as original FRNet)
    # =====================================================
    def frustum2pixel(self, frustum_features, coors, batch_size, stride=1):
        nx = self.nx // stride
        ny = self.ny // stride
        pixel_features = torch.zeros(
            (batch_size, ny, nx, frustum_features.shape[-1]),
            dtype=frustum_features.dtype, device=frustum_features.device)
        pixel_features[coors[:, 0], coors[:, 1], coors[:, 2]] = frustum_features
        pixel_features = pixel_features.permute(0, 3, 1, 2).contiguous()
        return pixel_features

    def pixel2point(self, pixel_features, coors, stride=1):
        pixel_features = pixel_features.permute(0, 2, 3, 1).contiguous()
        point_feats = pixel_features[coors[:, 0], coors[:, 1] // stride,
                                     coors[:, 2] // stride]
        return point_feats

    def point2frustum(self, point_features, pts_coors, stride=1):
        coors = pts_coors.clone()
        coors[:, 1] = pts_coors[:, 1] // stride
        coors[:, 2] = pts_coors[:, 2] // stride
        voxel_coors, inverse_map = torch.unique(
            coors, return_inverse=True, dim=0)
        frustum_features = torch_scatter.scatter_max(
            point_features, inverse_map, dim=0)[0]
        return voxel_coors, frustum_features

    # =====================================================
    # Forward pass
    # =====================================================
    def forward(self, voxel_dict: dict) -> dict:
        """Forward pass with explicit constraint branch.

        The original FRNet backbone forward is preserved. After backbone
        feature extraction, if images are available, the explicit constraint
        branch enhances the point features using image information.
        """
        # ---- Original FRNet backbone forward (unchanged) ----
        point_feats = voxel_dict['point_feats'][-1]
        voxel_feats = voxel_dict['voxel_feats']
        voxel_coors = voxel_dict['voxel_coors']
        pts_coors = voxel_dict['coors']
        batch_size = pts_coors[-1, 0].item() + 1
#===========================新增=============================
        if 'points' in voxel_dict:
            points = voxel_dict['points']
            if isinstance(points,list):
                voxel_dict['points_xyz']=torch.cat(
                    [p[:, :3] for p in points],dim=0)
            else:
                voxel_dict['points_xyz'] = points[:, :3]
#================================================================
        x = self.frustum2pixel(voxel_feats, voxel_coors, batch_size, stride=1)
        x = self.stem(x)
        map_point_feats = self.pixel2point(x, pts_coors, stride=1)
        fusion_point_feats = torch.cat((map_point_feats, point_feats), dim=1)
        point_feats = self.point_stem(fusion_point_feats)
        stride_voxel_coors, frustum_feats = self.point2frustum(
            point_feats, pts_coors, stride=1)
        pixel_feats = self.frustum2pixel(
            frustum_feats, stride_voxel_coors, batch_size, stride=1)
        fusion_pixel_feats = torch.cat((pixel_feats, x), dim=1)
        x = self.fusion_stem(fusion_pixel_feats)

        outs = [x]
        out_points = [point_feats]
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)

            # frustum-to-point fusion
            map_point_feats = self.pixel2point(
                x, pts_coors, stride=self.strides[i])
            fusion_point_feats = torch.cat(
                (map_point_feats, point_feats), dim=1)
            point_feats = self.point_fusion_layers[i](fusion_point_feats)

            # point-to-frustum fusion
            stride_voxel_coors, frustum_feats = self.point2frustum(
                point_feats, pts_coors, stride=self.strides[i])
            pixel_feats = self.frustum2pixel(
                frustum_feats, stride_voxel_coors, batch_size,
                stride=self.strides[i])
            fusion_pixel_feats = torch.cat((pixel_feats, x), dim=1)
            fuse_out = self.pixel_fusion_layers[i](fusion_pixel_feats)
            # residual-attentive
            attention_map = self.attention_layers[i](fuse_out)
            x = fuse_out * attention_map + x
            outs.append(x)
            out_points.append(point_feats)

        for i in range(len(outs)):
            if outs[i].shape != outs[0].shape:
                outs[i] = F.interpolate(
                    outs[i], size=outs[0].size()[2:],
                    mode='bilinear', align_corners=True)

        outs[0] = torch.cat(outs, dim=1)
        out_points[0] = torch.cat(out_points, dim=1)

        for layer_name, point_layer_name in zip(
                self.fuse_layers, self.point_fuse_layers):
            fuse_layer = getattr(self, layer_name)
            outs[0] = fuse_layer(outs[0])
            point_fuse_layer = getattr(self, point_layer_name)
            out_points[0] = point_fuse_layer(out_points[0])

        # ---- Explicit Constraint Branch (NEW) ----
        if self.enable_explicit and voxel_dict.get('has_images', False):
            # Extract image features
            images = voxel_dict['images']  # (B, 3, H_img, W_img)
            image_feat_map = self.image_backbone(images)  # (B, C_img, H', W')

#==============================新增==========================================
            voxel_dict['image_feat_map'] = image_feat_map
#==============================新增==========================================

            # Get projection coordinates
            proj_coords = voxel_dict['proj_coords']  # (N, 3) [batch, y, x]

            # Get final point features from backbone
            backbone_point_feats = out_points[0]  # (N, C_fuse)

            # Apply explicit constraint branch
            # This internally computes: offset prediction -> feature correction -> fusion
            # It also returns the corrected image features F'_image as intermediate result
            explicit_feats, corrected_image_feats = self.explicit_branch(
                voxel_feats=backbone_point_feats,
                image_feat_map=image_feat_map,
                proj_coords=proj_coords
            )  # (N, C_explicit), (N, C_align)

            # ---- Contrastive embeddings (for Loss_VI, training only) ----
            # φ_V: F_voxel -> Z_V (shared space)
            # φ_I: F'_image -> Z_I (shared space)
            if self.training:
                z_voxel = self.proj_head_voxel(backbone_point_feats)  # (N, D)
                z_image = self.proj_head_image(corrected_image_feats)  # (N, D)
                voxel_dict['z_voxel'] = z_voxel
                voxel_dict['z_image'] = z_image

            # Merge with original backbone output
            merged_point_feats = self.explicit_merge(
                torch.cat([backbone_point_feats, explicit_feats], dim=-1))
            out_points[0] = merged_point_feats

        # ---- Store results ----
        voxel_dict['voxel_feats'] = outs
        voxel_dict['point_feats_backbone'] = out_points
        return voxel_dict
