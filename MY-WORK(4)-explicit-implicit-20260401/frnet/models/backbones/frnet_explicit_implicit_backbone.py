"""
FRNet Backbone with Explicit + Implicit Constraint Branches.

This extends FRNetExplicitBackbone by adding the implicit constraint branch
based on 3D Gaussian Splatting. The implicit branch:

1. Takes the explicit-enhanced point features F_exp and point coordinates
2. Maps them to 3D Gaussian parameters via a lightweight MLP
3. Renders an implicit feature map via differentiable Gaussian splatting
4. The rendered feature map is stored in voxel_dict for loss computation

The implicit branch is ONLY active during training. At inference, it is
removed, but its regularization effect persists in the learned weights.

Architecture flow:
    Point Cloud → FRNet Backbone → F_u (point features)
    F_u + Image → Explicit Branch → F_exp (enhanced features)
    F_exp + xyz → Implicit Branch (MLP) → 3D Gaussians → Render → F̂_img
    Loss: L = L_seg + λ_exp·L_exp + λ_imp·||F̂_img - F_img||_1
"""

from typing import Optional, Sequence

import torch
import torch.nn as nn
from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig
from mmengine.model import BaseModule
from torch import Tensor

# Import the existing explicit backbone (we extend it)
from .frnet_explicit_backbone import FRNetExplicitBackbone
from .implicit_constraint import ImplicitConstraintBranch


@MODELS.register_module()
class FRNetExplicitImplicitBackbone(FRNetExplicitBackbone):
    """FRNet Backbone with both Explicit and Implicit Constraint Branches.

    Inherits all functionality from FRNetExplicitBackbone and adds the
    implicit constraint branch based on 3D Gaussian Splatting.

    Additional Args (compared to FRNetExplicitBackbone):
        enable_implicit (bool): Whether to enable implicit constraint.
            Default: True.
        implicit_feat_channels (int): Feature dimension for Gaussian splatting.
            Must match image encoder intermediate feature dimension. Default: 128.
        implicit_hidden_channels (int): MLP hidden layer dimension. Default: 128.
        implicit_num_mlp_layers (int): Number of MLP hidden layers. Default: 2.
        implicit_alpha_min (float): Minimum opacity for Gaussian pruning.
            Default: 0.01.
    """

    def __init__(self,
                 # ---- Original FRNet + Explicit params ----
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
                 image_backbone_cfg: Optional[dict] = None,
                 explicit_voxel_channels: int = 128,
                 explicit_image_channels: int = 128,
                 explicit_align_channels: int = 128,
                 explicit_out_channels: int = 128,
                 explicit_num_samples: int = 9,
                 enable_explicit: bool = True,
                 # ---- Implicit constraint params (NEW) ----
                 enable_implicit: bool = True,
                 implicit_feat_channels: int = 128,
                 implicit_hidden_channels: int = 128,
                 implicit_num_mlp_layers: int = 2,
                 implicit_alpha_min: float = 0.01,
                 init_cfg: OptMultiConfig = None) -> None:

        # Initialize the parent (FRNetExplicitBackbone)
        super().__init__(
            in_channels=in_channels,
            point_in_channels=point_in_channels,
            output_shape=output_shape,
            depth=depth,
            stem_channels=stem_channels,
            num_stages=num_stages,
            out_channels=out_channels,
            strides=strides,
            dilations=dilations,
            fuse_channels=fuse_channels,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            point_norm_cfg=point_norm_cfg,
            act_cfg=act_cfg,
            image_backbone_cfg=image_backbone_cfg,
            explicit_voxel_channels=explicit_voxel_channels,
            explicit_image_channels=explicit_image_channels,
            explicit_align_channels=explicit_align_channels,
            explicit_out_channels=explicit_out_channels,
            explicit_num_samples=explicit_num_samples,
            enable_explicit=enable_explicit,
            init_cfg=init_cfg,
        )

        self.enable_implicit = enable_implicit

        if enable_implicit:
            # The point feature dimension after explicit fusion
            # This is fuse_channels[-1] after explicit merge
            point_feat_dim = fuse_channels[-1]

            self.implicit_branch = ImplicitConstraintBranch(
                point_feat_channels=point_feat_dim,
                image_feat_channels=implicit_feat_channels,
                hidden_channels=implicit_hidden_channels,
                num_mlp_layers=implicit_num_mlp_layers,
                alpha_min=implicit_alpha_min,
            )

    def forward(self, voxel_dict: dict) -> dict:
        """Forward pass with explicit + implicit constraint branches.

        Calls the parent's forward (which includes explicit constraint),
        then if training and implicit is enabled, runs the implicit branch.

        The implicit branch:
        1. Takes the final point features (after explicit fusion) and xyz
        2. Runs them through the Gaussian MLP
        3. Renders to image plane via differentiable splatting
        4. Stores rendered feature map in voxel_dict for loss computation

        Args:
            voxel_dict (dict): Input dictionary with voxel/point data.

        Returns:
            dict: Updated voxel_dict with additional keys:
                - 'rendered_implicit_feat': (C, H', W') rendered feature map
                - 'image_feat_for_implicit': (C, H', W') image encoder features
                  (both only present during training with implicit enabled)
        """
        # ---- Run parent forward (FRNet backbone + explicit constraint) ----
        voxel_dict = super().forward(voxel_dict)

        # ---- Implicit Constraint Branch (training only) ----
        if (self.training and self.enable_implicit
                and voxel_dict.get('has_images', False)):

            # Get the enhanced point features after explicit fusion
            point_feats = voxel_dict['point_feats_backbone'][0]  # (N, C_fuse)

            # Get point coordinates in LiDAR frame
            # These are the original 3D coordinates of each point
            pts_coors = voxel_dict['coors']  # (N, 4) [batch_idx, z, y, x] or similar
            # We need the actual 3D xyz coordinates
            # They should be stored from the voxel encoder or data preprocessor
            xyz = voxel_dict.get('points_xyz', None)
            if xyz is None:
                # Fallback 1: try 'voxels' key which has per-point [x,y,z,intensity,...]
                # In FRNet, after frustum_region_group, voxel_dict['voxels'] is (N, 4+)
                voxels = voxel_dict.get('voxels', None)
                if voxels is not None and voxels.dim() == 2 and voxels.shape[-1] >= 3:
                    xyz = voxels[:, :3].clone()
                else:
                    # Fallback 2: try 'points' key
                    points = voxel_dict.get('points', None)
                    if points is not None:
                        if isinstance(points, list):
                            xyz = torch.cat([p[:, :3] for p in points], dim=0)
                        else:
                            xyz = points[:, :3]
                    else:
                        # Cannot run implicit branch without xyz
                        return voxel_dict

            # Get camera parameters
            viewmatrix = voxel_dict.get('viewmatrix', None)
            projmatrix = voxel_dict.get('projmatrix', None)

            if viewmatrix is None or projmatrix is None:
                # Try to construct from lidar2img
                lidar2img = voxel_dict.get('lidar2img', None)
                if lidar2img is None:
                    return voxel_dict

                # lidar2img is typically (3, 4) or (4, 4)
                # We need viewmatrix (4x4) and projmatrix (4x4)
                # For diff-gaussian-rasterization, projmatrix = lidar2img extended to 4x4
                if isinstance(lidar2img, torch.Tensor):
                    lidar2img_4x4 = torch.eye(4, device=lidar2img.device, dtype=lidar2img.dtype)
                    lidar2img_4x4[:lidar2img.shape[0], :lidar2img.shape[1]] = lidar2img
                else:
                    import numpy as np
                    lidar2img_np = np.eye(4)
                    lidar2img_np[:lidar2img.shape[0], :lidar2img.shape[1]] = lidar2img
                    lidar2img_4x4 = torch.tensor(lidar2img_np, device=point_feats.device,
                                                  dtype=torch.float32)

                # For the modified rasterizer that uses projmatrix directly:
                viewmatrix = torch.eye(4, device=point_feats.device, dtype=torch.float32)
                projmatrix = lidar2img_4x4.float()

            # Get image encoder feature map (from explicit branch)
            # The image backbone already extracted features during explicit forward
            image_feat_map = voxel_dict.get('image_feat_map', None)
            if image_feat_map is None:
                return voxel_dict

            # Image feature map dimensions
            _, C_img, H_feat, W_feat = image_feat_map.shape

            # Process each sample in the batch separately
            batch_size = pts_coors[:, 0].max().item() + 1
            rendered_feats_list = []
            image_feats_list = []

            for b in range(int(batch_size)):
                # Get points belonging to this batch
                batch_mask = (pts_coors[:, 0] == b)
                xyz_b = xyz[batch_mask]          # (N_b, 3)
                feats_b = point_feats[batch_mask]  # (N_b, C)

                if xyz_b.shape[0] == 0:
                    continue

                # Render implicit feature map for this sample
                rendered_feat = self.implicit_branch(
                    xyz=xyz_b,
                    point_feats=feats_b,
                    viewmatrix=viewmatrix,
                    projmatrix=projmatrix,
                    image_height=H_feat,
                    image_width=W_feat,
                )  # (C_feat, H_feat, W_feat)

                rendered_feats_list.append(rendered_feat)
                image_feats_list.append(image_feat_map[b])  # (C_img, H_feat, W_feat)

            if len(rendered_feats_list) > 0:
                # Stack batch
                rendered_feats = torch.stack(rendered_feats_list, dim=0)  # (B, C, H, W)
                image_feats = torch.stack(image_feats_list, dim=0)       # (B, C, H, W)

                voxel_dict['rendered_implicit_feat'] = rendered_feats
                voxel_dict['image_feat_for_implicit'] = image_feats

        return voxel_dict
