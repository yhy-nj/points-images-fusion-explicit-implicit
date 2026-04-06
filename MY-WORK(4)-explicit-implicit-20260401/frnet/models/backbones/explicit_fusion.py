# """
# Explicit Constraint Fusion Module for FRNet.
#
# This module implements the explicit constraint branch that performs:
# 1. 3D-to-2D projection using camera calibration
# 2. Offset prediction and feature correction (deformable sampling)
# 3. Cross-modal interaction with gated fusion (inspired by 2DPASS)
#
# Key components:
# - OffsetPredictionModule: Predicts projection offset Δp and importance weight Δm
# - FeatureCorrectionModule: Corrects image features via adaptive sampling
# - CrossModalFusionModule: 2DPASS-style learner + gated fusion
# - ExplicitConstraintBranch: Full explicit constraint pipeline
# """
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from mmcv.cnn import build_norm_layer, ConvModule
# from mmengine.model import BaseModule
# from torch import Tensor
# from typing import Optional, Tuple
#
#
# class OffsetPredictionModule(BaseModule):
#     """Predict projection offset Δp and importance weight Δm.
#
#     Given F_voxel (3D branch features mapped to point level) and
#     F_image (image features at projected positions), predict:
#     - Δp: 2D offset for the projected position (N, 2) or (N, K, 2) for K neighbors
#     - Δm: importance weight for each sampling position (N, K)
#
#     This handles the misalignment between 3D points and their 2D projections
#     caused by calibration errors, viewpoint differences, point cloud sparsity,
#     and semantic boundary ambiguity.
#     """
#
#     def __init__(self,
#                  voxel_channels: int,
#                  image_channels: int,
#                  num_samples: int = 9,
#                  norm_cfg: dict = dict(type='BN1d', eps=1e-3, momentum=0.01)):
#         super().__init__()
#         self.num_samples = num_samples  # 3x3 neighborhood = 9
#         in_channels = voxel_channels + image_channels
#
#         # Predict offset Δp for each sample point
#         self.offset_pred = nn.Sequential(
#             nn.Linear(in_channels, in_channels // 2, bias=False),
#             build_norm_layer(norm_cfg, in_channels // 2)[1],
#             nn.ReLU(inplace=True),
#             nn.Linear(in_channels // 2, num_samples * 2)  # (N, K*2) -> K offsets, each 2D
#         )
#
#         # Predict importance weight Δm for each sample point
#         self.weight_pred = nn.Sequential(
#             nn.Linear(in_channels, in_channels // 2, bias=False),
#             build_norm_layer(norm_cfg, in_channels // 2)[1],
#             nn.ReLU(inplace=True),
#             nn.Linear(in_channels // 2, num_samples)  # (N, K)
#         )
#
#         # Initialize offsets to zero, weights to uniform
#         nn.init.zeros_(self.offset_pred[-1].weight)
#         nn.init.zeros_(self.offset_pred[-1].bias)
#         nn.init.zeros_(self.weight_pred[-1].weight)
#         nn.init.zeros_(self.weight_pred[-1].bias)
#
#     def forward(self, voxel_feats: Tensor, image_feats: Tensor) -> Tuple[Tensor, Tensor]:
#         """
#         Args:
#             voxel_feats: (N, C_v) 3D voxel features at point level
#             image_feats: (N, C_i) image features at projected positions
#
#         Returns:
#             offsets: (N, K, 2) predicted offsets for K sampling positions
#             weights: (N, K) importance weights (softmax normalized)
#         """
#         combined = torch.cat([voxel_feats, image_feats], dim=-1)  # (N, C_v + C_i)
#
#         offsets = self.offset_pred(combined)  # (N, K*2)
#         offsets = offsets.view(-1, self.num_samples, 2)  # (N, K, 2)
#
#         weights = self.weight_pred(combined)  # (N, K)
#         weights = F.softmax(weights, dim=-1)  # normalize
#
#         return offsets, weights
#
#
# class FeatureCorrectionModule(BaseModule):
#     """Correct image features via adaptive offset sampling and weighted aggregation.
#
#     Given the projected 2D positions, predicted offsets Δp, importance weights Δm,
#     and the image feature map, perform:
#     1. For each point, compute K sampling positions: p + p_n + Δp_n
#        where p_n are base offsets from a 3x3 grid
#     2. Sample image features at these positions using bilinear interpolation
#     3. Weighted aggregate: F'_image = Σ w_n * F_image(p + p_n + Δp_n) * Δm_n
#
#     Formula (from Image 3, Eq. 3):
#     F'_image = Σ_{n=1}^{9} w_n · F_image(p + p_n + Δp_n) · Δm_n
#     """
#
#     def __init__(self,
#                  image_channels: int,
#                  out_channels: int,
#                  kernel_size: int = 3,
#                  norm_cfg: dict = dict(type='BN1d', eps=1e-3, momentum=0.01)):
#         super().__init__()
#         self.kernel_size = kernel_size
#         self.num_samples = kernel_size * kernel_size
#
#         # Generate base offsets for KxK grid (e.g., 3x3)
#         # p_n ∈ {(-1,-1), (-1,0), ..., (1,1)}
#         offsets = []
#         half = kernel_size // 2
#         for dy in range(-half, half + 1):
#             for dx in range(-half, half + 1):
#                 offsets.append([dx, dy])
#         self.register_buffer('base_offsets', torch.tensor(offsets, dtype=torch.float32))
#         # base_offsets: (K, 2)
#
#         # Learnable convolution weights w_n (like deformable conv)
#         self.conv_weights = nn.Parameter(torch.ones(self.num_samples) / self.num_samples)
#
#         # Post-correction projection
#         self.proj = nn.Sequential(
#             nn.Linear(image_channels, out_channels, bias=False),
#             build_norm_layer(norm_cfg, out_channels)[1],
#             nn.ReLU(inplace=True)
#         )
#
#     def forward(self,
#                 image_feat_map: Tensor,
#                 proj_coords: Tensor,
#                 offsets: Tensor,
#                 weights: Tensor) -> Tensor:
#         """
#         Args:
#             image_feat_map: (B, C, H_img, W_img) image feature map
#             proj_coords: (N, 3) [batch_idx, proj_y, proj_x] projected 2D positions
#             offsets: (N, K, 2) predicted offsets (dx, dy)
#             weights: (N, K) importance weights
#
#         Returns:
#             corrected_feats: (N, C_out) corrected image features
#         """
#         B, C, H, W = image_feat_map.shape
#         N = proj_coords.shape[0]
#         K = self.num_samples
#
#         batch_idx = proj_coords[:, 0].long()  # (N,)
#         base_y = proj_coords[:, 1]  # (N,)
#         base_x = proj_coords[:, 2]  # (N,)
#
#         # Compute sampling positions: (N, K, 2)
#         # base_offsets: (K, 2), offsets: (N, K, 2)
#         sample_offsets = self.base_offsets.unsqueeze(0) + offsets  # (N, K, 2)
#
#         # Absolute sampling positions
#         sample_x = base_x.unsqueeze(1) + sample_offsets[:, :, 0]  # (N, K)
#         sample_y = base_y.unsqueeze(1) + sample_offsets[:, :, 1]  # (N, K)
#
#         # Normalize to [-1, 1] for grid_sample
#         sample_x_norm = 2.0 * sample_x / (W - 1) - 1.0  # (N, K)
#         sample_y_norm = 2.0 * sample_y / (H - 1) - 1.0  # (N, K)
#
#         # Perform bilinear sampling per batch
#         # We need to handle variable number of points per batch
#         corrected_feats = torch.zeros(N, C, device=image_feat_map.device,
#                                       dtype=image_feat_map.dtype)
#
#         conv_w = F.softmax(self.conv_weights, dim=0)  # (K,)
#
#         for b in range(B):
#             mask = (batch_idx == b)  # (N_b,)
#             if mask.sum() == 0:
#                 continue
#
#             n_b = mask.sum()
#             sx = sample_x_norm[mask]  # (N_b, K)
#             sy = sample_y_norm[mask]  # (N_b, K)
#             w_b = weights[mask]  # (N_b, K)
#
#             # Create grid: (1, N_b, K, 2)
#             grid = torch.stack([sx, sy], dim=-1).unsqueeze(0)  # (1, N_b, K, 2)
#
#             # Sample: image_feat_map[b]: (C, H, W) -> (1, C, H, W)
#             feat_b = image_feat_map[b:b+1]  # (1, C, H, W)
#             # grid_sample expects (N, C, H_out, W_out) with grid (N, H_out, W_out, 2)
#             sampled = F.grid_sample(
#                 feat_b, grid, mode='bilinear', padding_mode='zeros',
#                 align_corners=True)  # (1, C, N_b, K)
#             sampled = sampled.squeeze(0).permute(1, 2, 0)  # (N_b, K, C)
#
#             # Weighted aggregation: w_n (conv weights) * sampled * Δm (importance weights)
#             # conv_w: (K,), w_b: (N_b, K), sampled: (N_b, K, C)
#             combined_weights = conv_w.unsqueeze(0) * w_b  # (N_b, K)
#             combined_weights = combined_weights.unsqueeze(-1)  # (N_b, K, 1)
#             agg_feats = (sampled * combined_weights).sum(dim=1)  # (N_b, C)
#
#             corrected_feats[mask] = agg_feats
#
#         # Project to output channels
#         corrected_feats = self.proj(corrected_feats)  # (N, C_out)
#         return corrected_feats
#
#
# class CrossModalFusionModule(BaseModule):
#     """Cross-modal fusion inspired by 2DPASS.
#
#     Pipeline:
#     1. Learner MLP: F_voxel -> F_learner (intermediate representation)
#     2. Enhanced 3D: F^e_3D = F_voxel + F_learner (residual)
#     3. Cross-modal interaction: H = MLP([F'_image; F_learner])
#     4. Gating: A = sigmoid(MLP_g(H))
#     5. Fused feature: F_fuse = F'_image + A ⊙ H
#     6. Output: F_out = MLP_out([F^e_3D; F_fuse])
#     """
#
#     def __init__(self,
#                  voxel_channels: int,
#                  image_channels: int,
#                  out_channels: int,
#                  norm_cfg: dict = dict(type='BN1d', eps=1e-3, momentum=0.01)):
#         super().__init__()
#
#         # Learner MLP: maps F_voxel to cross-modal intermediate
#         self.learner_mlp = nn.Sequential(
#             nn.Linear(voxel_channels, voxel_channels, bias=False),
#             build_norm_layer(norm_cfg, voxel_channels)[1],
#             nn.ReLU(inplace=True)
#         )
#
#         # Fusion MLP: concatenate F'_image and F_learner -> H
#         fusion_in = image_channels + voxel_channels
#         self.fusion_mlp = nn.Sequential(
#             nn.Linear(fusion_in, voxel_channels, bias=False),
#             build_norm_layer(norm_cfg, voxel_channels)[1],
#             nn.ReLU(inplace=True)
#         )
#
#         # Gate MLP: generate adaptive weight A from H
#         self.gate_mlp = nn.Sequential(
#             nn.Linear(voxel_channels, voxel_channels, bias=False),
#             build_norm_layer(norm_cfg, voxel_channels)[1],
#             nn.Sigmoid()
#         )
#
#         # Output MLP: concatenate F^e_3D and F_fuse -> F_out
#         self.output_mlp = nn.Sequential(
#             nn.Linear(voxel_channels + image_channels, out_channels, bias=False),
#             build_norm_layer(norm_cfg, out_channels)[1],
#             nn.ReLU(inplace=True)
#         )
#
#     def forward(self, voxel_feats: Tensor, corrected_image_feats: Tensor) -> Tensor:
#         """
#         Args:
#             voxel_feats: (N, C_v) 3D branch voxel features at point level
#             corrected_image_feats: (N, C_i) corrected image features F'_image
#
#         Returns:
#             fused_feats: (N, C_out) output fused features
#
#         Note: This is the V1 version. Prefer CrossModalFusionModuleV2 which
#               has cleaner dimension handling.
#         """
#         # Learner MLP
#         f_learner = self.learner_mlp(voxel_feats)  # (N, C_v)
#
#         # Enhanced 3D features (residual)
#         f_3d_enhanced = voxel_feats + f_learner  # (N, C_v)
#
#         # Cross-modal interaction: H has C_v channels
#         h = self.fusion_mlp(
#             torch.cat([corrected_image_feats, f_learner], dim=-1))  # (N, C_v)
#
#         # Gating
#         a = self.gate_mlp(h)  # (N, C_v)
#
#         # Output: concat enhanced 3D and image, then add gated info
#         f_out = self.output_mlp(
#             torch.cat([f_3d_enhanced, corrected_image_feats], dim=-1))  # (N, C_out)
#
#         return f_out
#
#
# class CrossModalFusionModuleV2(BaseModule):
#     """Cleaner version of cross-modal fusion following Image 1 exactly.
#
#     F_fuse = F'_image + A ⊙ H
#     where:
#         F_learner = MLP_learner(F_voxel)
#         F^e_3D = F_voxel + F_learner
#         H = MLP_fusion([F'_image; F_learner])  -- cross-modal interaction
#         A = sigmoid(MLP_g(H))  -- gating weight
#         F_fuse = F'_image + A ⊙ H  -- gated fusion (image-side enhanced)
#         F_out = MLP_out([F^e_3D; F_fuse])  -- final output
#     """
#
#     def __init__(self,
#                  voxel_channels: int,
#                  image_channels: int,
#                  out_channels: int,
#                  norm_cfg: dict = dict(type='BN1d', eps=1e-3, momentum=0.01)):
#         super().__init__()
#
#         # Learner MLP
#         self.learner_mlp = nn.Sequential(
#             nn.Linear(voxel_channels, voxel_channels, bias=False),
#             build_norm_layer(norm_cfg, voxel_channels)[1],
#             nn.ReLU(inplace=True)
#         )
#
#         # Fusion MLP: [F'_image; F_learner] -> H (same dim as image_channels)
#         self.fusion_mlp = nn.Sequential(
#             nn.Linear(image_channels + voxel_channels, image_channels, bias=False),
#             build_norm_layer(norm_cfg, image_channels)[1],
#             nn.ReLU(inplace=True)
#         )
#
#         # Gate MLP: H -> A ∈ (0,1)
#         self.gate_mlp = nn.Sequential(
#             nn.Linear(image_channels, image_channels, bias=False),
#             build_norm_layer(norm_cfg, image_channels)[1],
#             nn.Sigmoid()
#         )
#
#         # Output MLP: [F^e_3D; F_fuse] -> F_out
#         self.output_mlp = nn.Sequential(
#             nn.Linear(voxel_channels + image_channels, out_channels, bias=False),
#             build_norm_layer(norm_cfg, out_channels)[1],
#             nn.ReLU(inplace=True)
#         )
#
#     def forward(self, voxel_feats: Tensor, corrected_image_feats: Tensor) -> Tensor:
#         """
#         Args:
#             voxel_feats: (N, C_v) 3D voxel features
#             corrected_image_feats: (N, C_i) corrected image features F'_image
#
#         Returns:
#             f_out: (N, C_out) fused output features
#         """
#         # Learner: project voxel to cross-modal space
#         f_learner = self.learner_mlp(voxel_feats)  # (N, C_v)
#
#         # Enhanced 3D: residual
#         f_3d_enhanced = voxel_feats + f_learner  # (N, C_v)
#
#         # Cross-modal interaction
#         h = self.fusion_mlp(
#             torch.cat([corrected_image_feats, f_learner], dim=-1))  # (N, C_i)
#
#         # Gating
#         a = self.gate_mlp(h)  # (N, C_i)
#
#         # Gated fusion (image-side)
#         f_fuse = corrected_image_feats + a * h  # (N, C_i)
#
#         # Output
#         f_out = self.output_mlp(
#             torch.cat([f_3d_enhanced, f_fuse], dim=-1))  # (N, C_out)
#
#         return f_out
#
#
# class ExplicitConstraintBranch(BaseModule):
#     """Complete explicit constraint branch.
#
#     This module takes:
#     - voxel_feats: (N, C_v) point-level 3D features from FRNet backbone
#     - image_feat_map: (B, C_img, H_img, W_img) image feature map from image backbone
#     - proj_coords: (N, 3) [batch_idx, proj_y, proj_x] 2D projection coordinates
#
#     And produces:
#     - enhanced_feats: (N, C_out) enhanced point features incorporating image info
#
#     Steps:
#     1. Extract F_image at projected positions
#     2. MLP to align F_image dimensions with 3D branch
#     3. Offset prediction: (F_voxel, F_image) -> (Δp, Δm)
#     4. Feature correction: sample and aggregate -> F'_image
#     5. Cross-modal fusion: (F_voxel, F'_image) -> F_out
#     """
#
#     def __init__(self,
#                  voxel_channels: int,
#                  image_channels: int,
#                  image_align_channels: int,
#                  out_channels: int,
#                  num_samples: int = 9,
#                  norm_cfg: dict = dict(type='BN1d', eps=1e-3, momentum=0.01)):
#         super().__init__()
#
#         self.voxel_channels = voxel_channels
#         self.image_channels = image_channels
#         self.image_align_channels = image_align_channels
#
#         # MLP to project raw image features to aligned dimension
#         self.image_align_mlp = nn.Sequential(
#             nn.Linear(image_channels, image_align_channels, bias=False),
#             build_norm_layer(norm_cfg, image_align_channels)[1],
#             nn.ReLU(inplace=True)
#         )
#
#         # Offset prediction
#         self.offset_pred = OffsetPredictionModule(
#             voxel_channels=voxel_channels,
#             image_channels=image_align_channels,
#             num_samples=num_samples,
#             norm_cfg=norm_cfg
#         )
#
#         # Feature correction
#         self.feat_correction = FeatureCorrectionModule(
#             image_channels=image_channels,  # raw image channels for sampling
#             out_channels=image_align_channels,
#             kernel_size=int(num_samples ** 0.5) if int(num_samples ** 0.5) ** 2 == num_samples else 3,
#             norm_cfg=norm_cfg
#         )
#
#         # Cross-modal fusion
#         self.cross_modal_fusion = CrossModalFusionModuleV2(
#             voxel_channels=voxel_channels,
#             image_channels=image_align_channels,
#             out_channels=out_channels,
#             norm_cfg=norm_cfg
#         )
#
#     def _sample_image_feats(self, image_feat_map: Tensor,
#                             proj_coords: Tensor) -> Tensor:
#         """Sample image features at projected positions.
#
#         Args:
#             image_feat_map: (B, C, H, W)
#             proj_coords: (N, 3) [batch_idx, y, x]
#
#         Returns:
#             sampled_feats: (N, C)
#         """
#         B, C, H, W = image_feat_map.shape
#         N = proj_coords.shape[0]
#
#         sampled = torch.zeros(N, C, device=image_feat_map.device,
#                               dtype=image_feat_map.dtype)
#         batch_idx = proj_coords[:, 0].long()
#         y = proj_coords[:, 1]
#         x = proj_coords[:, 2]
#
#         # Normalize coordinates for grid_sample
#         x_norm = 2.0 * x / (W - 1) - 1.0
#         y_norm = 2.0 * y / (H - 1) - 1.0
#
#         for b in range(B):
#             mask = (batch_idx == b)
#             if mask.sum() == 0:
#                 continue
#             n_b = mask.sum()
#             grid = torch.stack([x_norm[mask], y_norm[mask]], dim=-1)  # (N_b, 2)
#             grid = grid.unsqueeze(0).unsqueeze(2)  # (1, N_b, 1, 2)
#
#             feat_b = image_feat_map[b:b+1]  # (1, C, H, W)
#             s = F.grid_sample(feat_b, grid, mode='bilinear',
#                               padding_mode='zeros', align_corners=True)
#             # s: (1, C, N_b, 1)
#             sampled[mask] = s.squeeze(0).squeeze(-1).permute(1, 0)  # (N_b, C)
#
#         return sampled
#
#     def forward(self,
#                 voxel_feats: Tensor,
#                 image_feat_map: Tensor,
#                 proj_coords: Tensor) -> Tensor:
#         """
#         Args:
#             voxel_feats: (N, C_v) point-level 3D features
#             image_feat_map: (B, C_img, H_img, W_img) image feature map
#             proj_coords: (N, 3) [batch_idx, proj_y, proj_x]
#
#         Returns:
#             enhanced_feats: (N, C_out) enhanced features
#         """
#         # Step 1: Sample image features at projected positions
#         raw_image_feats = self._sample_image_feats(image_feat_map, proj_coords)
#         # raw_image_feats: (N, C_img)
#
#         # Step 2: Align image feature dimension
#         aligned_image_feats = self.image_align_mlp(raw_image_feats)
#         # aligned_image_feats: (N, C_align)
#
#         # Step 3: Offset prediction
#         offsets, weights = self.offset_pred(voxel_feats, aligned_image_feats)
#         # offsets: (N, K, 2), weights: (N, K)
#
#         # Step 4: Feature correction (deformable sampling + weighted aggregation)
#         corrected_image_feats = self.feat_correction(
#             image_feat_map, proj_coords, offsets, weights)
#         # corrected_image_feats: (N, C_align)
#
#         # Step 5: Cross-modal fusion
#         enhanced_feats = self.cross_modal_fusion(voxel_feats, corrected_image_feats)
#         # enhanced_feats: (N, C_out)
#
#         return enhanced_feats
"""



Explicit Constraint Fusion Module for FRNet.

This module implements the explicit constraint branch that performs:
1. 3D-to-2D projection using camera calibration
2. Offset prediction and feature correction (deformable sampling)
3. Cross-modal interaction with gated fusion (inspired by 2DPASS)

Key components:
- OffsetPredictionModule: Predicts projection offset Δp and importance weight Δm
- FeatureCorrectionModule: Corrects image features via adaptive sampling
- CrossModalFusionModule: 2DPASS-style learner + gated fusion
- ExplicitConstraintBranch: Full explicit constraint pipeline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer, ConvModule
from mmengine.model import BaseModule
from torch import Tensor
from typing import Optional, Tuple


class OffsetPredictionModule(BaseModule):
    """Predict projection offset Δp and importance weight Δm.

    Given F_voxel (3D branch features mapped to point level) and
    F_image (image features at projected positions), predict:
    - Δp: 2D offset for the projected position (N, 2) or (N, K, 2) for K neighbors
    - Δm: importance weight for each sampling position (N, K)

    This handles the misalignment between 3D points and their 2D projections
    caused by calibration errors, viewpoint differences, point cloud sparsity,
    and semantic boundary ambiguity.
    """

    def __init__(self,
                 voxel_channels: int,
                 image_channels: int,
                 num_samples: int = 9,
                 norm_cfg: dict = dict(type='BN1d', eps=1e-3, momentum=0.01)):
        super().__init__()
        self.num_samples = num_samples  # 3x3 neighborhood = 9
        in_channels = voxel_channels + image_channels

        # Predict offset Δp for each sample point
        self.offset_pred = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2, bias=False),
            build_norm_layer(norm_cfg, in_channels // 2)[1],
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 2, num_samples * 2)  # (N, K*2) -> K offsets, each 2D
        )

        # Predict importance weight Δm for each sample point
        self.weight_pred = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2, bias=False),
            build_norm_layer(norm_cfg, in_channels // 2)[1],
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 2, num_samples)  # (N, K)
        )

        # Initialize offsets to zero, weights to uniform
        nn.init.zeros_(self.offset_pred[-1].weight)
        nn.init.zeros_(self.offset_pred[-1].bias)
        nn.init.zeros_(self.weight_pred[-1].weight)
        nn.init.zeros_(self.weight_pred[-1].bias)

    def forward(self, voxel_feats: Tensor, image_feats: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            voxel_feats: (N, C_v) 3D voxel features at point level
            image_feats: (N, C_i) image features at projected positions

        Returns:
            offsets: (N, K, 2) predicted offsets for K sampling positions
            weights: (N, K) importance weights (softmax normalized)
        """
        combined = torch.cat([voxel_feats, image_feats], dim=-1)  # (N, C_v + C_i)

        offsets = self.offset_pred(combined)  # (N, K*2)
        offsets = offsets.view(-1, self.num_samples, 2)  # (N, K, 2)

        weights = self.weight_pred(combined)  # (N, K)
        weights = F.softmax(weights, dim=-1)  # normalize

        return offsets, weights


class FeatureCorrectionModule(BaseModule):
    """Correct image features via adaptive offset sampling and weighted aggregation.

    Given the projected 2D positions, predicted offsets Δp, importance weights Δm,
    and the image feature map, perform:
    1. For each point, compute K sampling positions: p + p_n + Δp_n
       where p_n are base offsets from a 3x3 grid
    2. Sample image features at these positions using bilinear interpolation
    3. Weighted aggregate: F'_image = Σ w_n * F_image(p + p_n + Δp_n) * Δm_n

    Formula (from Image 3, Eq. 3):
    F'_image = Σ_{n=1}^{9} w_n · F_image(p + p_n + Δp_n) · Δm_n
    """

    def __init__(self,
                 image_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 norm_cfg: dict = dict(type='BN1d', eps=1e-3, momentum=0.01)):
        super().__init__()
        self.kernel_size = kernel_size
        self.num_samples = kernel_size * kernel_size

        # Generate base offsets for KxK grid (e.g., 3x3)
        # p_n ∈ {(-1,-1), (-1,0), ..., (1,1)}
        offsets = []
        half = kernel_size // 2
        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                offsets.append([dx, dy])
        self.register_buffer('base_offsets', torch.tensor(offsets, dtype=torch.float32))
        # base_offsets: (K, 2)

        # Learnable convolution weights w_n (like deformable conv)
        self.conv_weights = nn.Parameter(torch.ones(self.num_samples) / self.num_samples)

        # Post-correction projection
        self.proj = nn.Sequential(
            nn.Linear(image_channels, out_channels, bias=False),
            build_norm_layer(norm_cfg, out_channels)[1],
            nn.ReLU(inplace=True)
        )

    def forward(self,
                image_feat_map: Tensor,
                proj_coords: Tensor,
                offsets: Tensor,
                weights: Tensor) -> Tensor:
        """
        Args:
            image_feat_map: (B, C, H_img, W_img) image feature map
            proj_coords: (N, 3) [batch_idx, proj_y, proj_x] projected 2D positions
            offsets: (N, K, 2) predicted offsets (dx, dy)
            weights: (N, K) importance weights

        Returns:
            corrected_feats: (N, C_out) corrected image features
        """
        B, C, H, W = image_feat_map.shape
        N = proj_coords.shape[0]
        K = self.num_samples

        batch_idx = proj_coords[:, 0].long()  # (N,)
        base_y = proj_coords[:, 1]  # (N,)
        base_x = proj_coords[:, 2]  # (N,)

        # Compute sampling positions: (N, K, 2)
        # base_offsets: (K, 2), offsets: (N, K, 2)
        sample_offsets = self.base_offsets.unsqueeze(0) + offsets  # (N, K, 2)

        # Absolute sampling positions
        sample_x = base_x.unsqueeze(1) + sample_offsets[:, :, 0]  # (N, K)
        sample_y = base_y.unsqueeze(1) + sample_offsets[:, :, 1]  # (N, K)

        # Normalize to [-1, 1] for grid_sample
        sample_x_norm = 2.0 * sample_x / (W - 1) - 1.0  # (N, K)
        sample_y_norm = 2.0 * sample_y / (H - 1) - 1.0  # (N, K)

        # Perform bilinear sampling per batch
        # We need to handle variable number of points per batch
        corrected_feats = torch.zeros(N, C, device=image_feat_map.device,
                                      dtype=image_feat_map.dtype)

        conv_w = F.softmax(self.conv_weights, dim=0)  # (K,)

        for b in range(B):
            mask = (batch_idx == b)  # (N_b,)
            if mask.sum() == 0:
                continue

            n_b = mask.sum()
            sx = sample_x_norm[mask]  # (N_b, K)
            sy = sample_y_norm[mask]  # (N_b, K)
            w_b = weights[mask]  # (N_b, K)

            # Create grid: (1, N_b, K, 2)
            grid = torch.stack([sx, sy], dim=-1).unsqueeze(0)  # (1, N_b, K, 2)

            # Sample: image_feat_map[b]: (C, H, W) -> (1, C, H, W)
            feat_b = image_feat_map[b:b+1]  # (1, C, H, W)
            # grid_sample expects (N, C, H_out, W_out) with grid (N, H_out, W_out, 2)
            sampled = F.grid_sample(
                feat_b, grid, mode='bilinear', padding_mode='zeros',
                align_corners=True)  # (1, C, N_b, K)
            sampled = sampled.squeeze(0).permute(1, 2, 0)  # (N_b, K, C)

            # Weighted aggregation: w_n (conv weights) * sampled * Δm (importance weights)
            # conv_w: (K,), w_b: (N_b, K), sampled: (N_b, K, C)
            combined_weights = conv_w.unsqueeze(0) * w_b  # (N_b, K)
            combined_weights = combined_weights.unsqueeze(-1)  # (N_b, K, 1)
            agg_feats = (sampled * combined_weights).sum(dim=1)  # (N_b, C)

            corrected_feats[mask] = agg_feats

        # Project to output channels
        corrected_feats = self.proj(corrected_feats)  # (N, C_out)
        return corrected_feats


class CrossModalFusionModule(BaseModule):
    """Cross-modal fusion inspired by 2DPASS.

    Pipeline:
    1. Learner MLP: F_voxel -> F_learner (intermediate representation)
    2. Enhanced 3D: F^e_3D = F_voxel + F_learner (residual)
    3. Cross-modal interaction: H = MLP([F'_image; F_learner])
    4. Gating: A = sigmoid(MLP_g(H))
    5. Fused feature: F_fuse = F'_image + A ⊙ H
    6. Output: F_out = MLP_out([F^e_3D; F_fuse])
    """

    def __init__(self,
                 voxel_channels: int,
                 image_channels: int,
                 out_channels: int,
                 norm_cfg: dict = dict(type='BN1d', eps=1e-3, momentum=0.01)):
        super().__init__()

        # Learner MLP: maps F_voxel to cross-modal intermediate
        self.learner_mlp = nn.Sequential(
            nn.Linear(voxel_channels, voxel_channels, bias=False),
            build_norm_layer(norm_cfg, voxel_channels)[1],
            nn.ReLU(inplace=True)
        )

        # Fusion MLP: concatenate F'_image and F_learner -> H
        fusion_in = image_channels + voxel_channels
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_in, voxel_channels, bias=False),
            build_norm_layer(norm_cfg, voxel_channels)[1],
            nn.ReLU(inplace=True)
        )

        # Gate MLP: generate adaptive weight A from H
        self.gate_mlp = nn.Sequential(
            nn.Linear(voxel_channels, voxel_channels, bias=False),
            build_norm_layer(norm_cfg, voxel_channels)[1],
            nn.Sigmoid()
        )

        # Output MLP: concatenate F^e_3D and F_fuse -> F_out
        self.output_mlp = nn.Sequential(
            nn.Linear(voxel_channels + image_channels, out_channels, bias=False),
            build_norm_layer(norm_cfg, out_channels)[1],
            nn.ReLU(inplace=True)
        )

    def forward(self, voxel_feats: Tensor, corrected_image_feats: Tensor) -> Tensor:
        """
        Args:
            voxel_feats: (N, C_v) 3D branch voxel features at point level
            corrected_image_feats: (N, C_i) corrected image features F'_image

        Returns:
            fused_feats: (N, C_out) output fused features

        Note: This is the V1 version. Prefer CrossModalFusionModuleV2 which
              has cleaner dimension handling.
        """
        # Learner MLP
        f_learner = self.learner_mlp(voxel_feats)  # (N, C_v)

        # Enhanced 3D features (residual)
        f_3d_enhanced = voxel_feats + f_learner  # (N, C_v)

        # Cross-modal interaction: H has C_v channels
        h = self.fusion_mlp(
            torch.cat([corrected_image_feats, f_learner], dim=-1))  # (N, C_v)

        # Gating
        a = self.gate_mlp(h)  # (N, C_v)

        # Output: concat enhanced 3D and image, then add gated info
        f_out = self.output_mlp(
            torch.cat([f_3d_enhanced, corrected_image_feats], dim=-1))  # (N, C_out)

        return f_out


class CrossModalFusionModuleV2(BaseModule):
    """Cleaner version of cross-modal fusion following Image 1 exactly.

    F_fuse = F'_image + A ⊙ H
    where:
        F_learner = MLP_learner(F_voxel)
        F^e_3D = F_voxel + F_learner
        H = MLP_fusion([F'_image; F_learner])  -- cross-modal interaction
        A = sigmoid(MLP_g(H))  -- gating weight
        F_fuse = F'_image + A ⊙ H  -- gated fusion (image-side enhanced)
        F_out = MLP_out([F^e_3D; F_fuse])  -- final output
    """

    def __init__(self,
                 voxel_channels: int,
                 image_channels: int,
                 out_channels: int,
                 norm_cfg: dict = dict(type='BN1d', eps=1e-3, momentum=0.01)):
        super().__init__()

        # Learner MLP
        self.learner_mlp = nn.Sequential(
            nn.Linear(voxel_channels, voxel_channels, bias=False),
            build_norm_layer(norm_cfg, voxel_channels)[1],
            nn.ReLU(inplace=True)
        )

        # Fusion MLP: [F'_image; F_learner] -> H (same dim as image_channels)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(image_channels + voxel_channels, image_channels, bias=False),
            build_norm_layer(norm_cfg, image_channels)[1],
            nn.ReLU(inplace=True)
        )

        # Gate MLP: H -> A ∈ (0,1)
        self.gate_mlp = nn.Sequential(
            nn.Linear(image_channels, image_channels, bias=False),
            build_norm_layer(norm_cfg, image_channels)[1],
            nn.Sigmoid()
        )

        # Output MLP: [F^e_3D; F_fuse] -> F_out
        self.output_mlp = nn.Sequential(
            nn.Linear(voxel_channels + image_channels, out_channels, bias=False),
            build_norm_layer(norm_cfg, out_channels)[1],
            nn.ReLU(inplace=True)
        )

    def forward(self, voxel_feats: Tensor, corrected_image_feats: Tensor) -> Tensor:
        """
        Args:
            voxel_feats: (N, C_v) 3D voxel features
            corrected_image_feats: (N, C_i) corrected image features F'_image

        Returns:
            f_out: (N, C_out) fused output features
        """
        # Learner: project voxel to cross-modal space
        f_learner = self.learner_mlp(voxel_feats)  # (N, C_v)

        # Enhanced 3D: residual
        f_3d_enhanced = voxel_feats + f_learner  # (N, C_v)

        # Cross-modal interaction
        h = self.fusion_mlp(
            torch.cat([corrected_image_feats, f_learner], dim=-1))  # (N, C_i)

        # Gating
        a = self.gate_mlp(h)  # (N, C_i)

        # Gated fusion (image-side)
        f_fuse = corrected_image_feats + a * h  # (N, C_i)

        # Output
        f_out = self.output_mlp(
            torch.cat([f_3d_enhanced, f_fuse], dim=-1))  # (N, C_out)

        return f_out


class ExplicitConstraintBranch(BaseModule):
    """Complete explicit constraint branch.

    This module takes:
    - voxel_feats: (N, C_v) point-level 3D features from FRNet backbone
    - image_feat_map: (B, C_img, H_img, W_img) image feature map from image backbone
    - proj_coords: (N, 3) [batch_idx, proj_y, proj_x] 2D projection coordinates

    And produces:
    - enhanced_feats: (N, C_out) enhanced point features incorporating image info

    Steps:
    1. Extract F_image at projected positions
    2. MLP to align F_image dimensions with 3D branch
    3. Offset prediction: (F_voxel, F_image) -> (Δp, Δm)
    4. Feature correction: sample and aggregate -> F'_image
    5. Cross-modal fusion: (F_voxel, F'_image) -> F_out
    """

    def __init__(self,
                 voxel_channels: int,
                 image_channels: int,
                 image_align_channels: int,
                 out_channels: int,
                 num_samples: int = 9,
                 norm_cfg: dict = dict(type='BN1d', eps=1e-3, momentum=0.01)):
        super().__init__()

        self.voxel_channels = voxel_channels
        self.image_channels = image_channels
        self.image_align_channels = image_align_channels

        # MLP to project raw image features to aligned dimension
        self.image_align_mlp = nn.Sequential(
            nn.Linear(image_channels, image_align_channels, bias=False),
            build_norm_layer(norm_cfg, image_align_channels)[1],
            nn.ReLU(inplace=True)
        )

        # Offset prediction
        self.offset_pred = OffsetPredictionModule(
            voxel_channels=voxel_channels,
            image_channels=image_align_channels,
            num_samples=num_samples,
            norm_cfg=norm_cfg
        )

        # Feature correction
        self.feat_correction = FeatureCorrectionModule(
            image_channels=image_channels,  # raw image channels for sampling
            out_channels=image_align_channels,
            kernel_size=int(num_samples ** 0.5) if int(num_samples ** 0.5) ** 2 == num_samples else 3,
            norm_cfg=norm_cfg
        )

        # Cross-modal fusion
        self.cross_modal_fusion = CrossModalFusionModuleV2(
            voxel_channels=voxel_channels,
            image_channels=image_align_channels,
            out_channels=out_channels,
            norm_cfg=norm_cfg
        )

    def _sample_image_feats(self, image_feat_map: Tensor,
                            proj_coords: Tensor) -> Tensor:
        """Sample image features at projected positions.

        Args:
            image_feat_map: (B, C, H, W)
            proj_coords: (N, 3) [batch_idx, y, x]

        Returns:
            sampled_feats: (N, C)
        """
        B, C, H, W = image_feat_map.shape
        N = proj_coords.shape[0]

        sampled = torch.zeros(N, C, device=image_feat_map.device,
                              dtype=image_feat_map.dtype)
        batch_idx = proj_coords[:, 0].long()
        y = proj_coords[:, 1]
        x = proj_coords[:, 2]

        # Normalize coordinates for grid_sample
        x_norm = 2.0 * x / (W - 1) - 1.0
        y_norm = 2.0 * y / (H - 1) - 1.0

        for b in range(B):
            mask = (batch_idx == b)
            if mask.sum() == 0:
                continue
            n_b = mask.sum()
            grid = torch.stack([x_norm[mask], y_norm[mask]], dim=-1)  # (N_b, 2)
            grid = grid.unsqueeze(0).unsqueeze(2)  # (1, N_b, 1, 2)

            feat_b = image_feat_map[b:b+1]  # (1, C, H, W)
            s = F.grid_sample(feat_b, grid, mode='bilinear',
                              padding_mode='zeros', align_corners=True)
            # s: (1, C, N_b, 1)
            sampled[mask] = s.squeeze(0).squeeze(-1).permute(1, 0)  # (N_b, C)

        return sampled

    def forward(self,
                voxel_feats: Tensor,
                image_feat_map: Tensor,
                proj_coords: Tensor):
        """
        Args:
            voxel_feats: (N, C_v) point-level 3D features
            image_feat_map: (B, C_img, H_img, W_img) image feature map
            proj_coords: (N, 3) [batch_idx, proj_y, proj_x]

        Returns:
            enhanced_feats: (N, C_out) enhanced features from cross-modal fusion
            corrected_image_feats: (N, C_align) corrected image features F'_image
                (used by contrastive loss)
        """
        # Step 1: Sample image features at projected positions
        raw_image_feats = self._sample_image_feats(image_feat_map, proj_coords)
        # raw_image_feats: (N, C_img)

        # Step 2: Align image feature dimension
        aligned_image_feats = self.image_align_mlp(raw_image_feats)
        # aligned_image_feats: (N, C_align)

        # Step 3: Offset prediction
        offsets, weights = self.offset_pred(voxel_feats, aligned_image_feats)
        # offsets: (N, K, 2), weights: (N, K)

        # Step 4: Feature correction (deformable sampling + weighted aggregation)
        corrected_image_feats = self.feat_correction(
            image_feat_map, proj_coords, offsets, weights)
        # corrected_image_feats: (N, C_align)

        # Step 5: Cross-modal fusion
        enhanced_feats = self.cross_modal_fusion(voxel_feats, corrected_image_feats)
        # enhanced_feats: (N, C_out)

        return enhanced_feats, corrected_image_feats