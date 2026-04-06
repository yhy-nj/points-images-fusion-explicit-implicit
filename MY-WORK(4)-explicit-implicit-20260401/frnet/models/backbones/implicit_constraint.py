# """
# Implicit Constraint Branch based on 3D Gaussian Splatting.
#
# This module implements the implicit constraint branch that:
# 1. Maps each point's coordinates + explicit-enhanced features to 3D Gaussian parameters
#    via a lightweight MLP (Points-to-Gaussians).
# 2. Performs differentiable Gaussian Splatting to render an implicit feature map.
# 3. The rendered feature map is supervised against image encoder features (L_imp).
#
# The branch is ONLY active during training. At inference, it is removed,
# but the regularization it imposed on the backbone persists in the learned weights.
#
# Key components:
# - PointsToGaussiansMLP: Predicts (μ_offset, Σ_compact, α, e) per point.
# - ImplicitConstraintBranch: Orchestrates MLP + differentiable rendering.
#
# References:
# - GaussianCaR (Montiel-Marín et al., 2025): Points-to-Gaussians module,
#   coarse-to-fine positioning, compact covariance representation.
# - 3D Gaussian Splatting (Kerbl et al., 2023): Differentiable rasterization.
# """
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from mmengine.model import BaseModule
# from torch import Tensor
# from typing import Optional, Tuple
#
# try:
#     from diff_gaussian_rasterization import (
#         GaussianRasterizationSettings,
#         GaussianRasterizer,
#     )
#     HAS_DIFF_GAUSSIAN_RASTERIZATION = True
# except ImportError:
#     HAS_DIFF_GAUSSIAN_RASTERIZATION = False
#
#
# class PointsToGaussiansMLP(BaseModule):
#     """MLP that maps point coordinates + features to 3D Gaussian parameters.
#
#     Input:  concat(xyz_i, F_exp_i)  →  shape (N, 3 + C_in)
#     Output: (offset_3d, cov_compact_6d, opacity_1d, implicit_feat)
#
#     Following GaussianCaR's Points-to-Gaussians design:
#     - Position: coarse-to-fine, μ_i = xyz_i + Δp_i (MLP predicts offset only)
#     - Covariance: compact 6D representation R_cov ∈ R^6 = [xx,xy,xz,yy,yz,zz],
#       eigenvalues enforced positive via softplus → size & orientation
#     - Opacity: sigmoid → [0, 1], with α_min = 0.01 threshold
#     - Implicit feature: e_i ∈ R^{C_feat}, aligned with image encoder feature dim
#
#     Args:
#         in_channels (int): Input feature dimension (3 + C_exp).
#         hidden_channels (int): Hidden layer dimension. Default: 128.
#         feat_channels (int): Output implicit feature dimension (must match
#             image encoder intermediate feature dim). Default: 128.
#         num_layers (int): Number of hidden layers. Default: 2.
#         alpha_min (float): Minimum opacity threshold. Default: 0.01.
#     """
#
#     def __init__(self,
#                  in_channels: int,
#                  hidden_channels: int = 128,
#                  feat_channels: int = 128,
#                  num_layers: int = 2,
#                  alpha_min: float = 0.01) -> None:
#         super().__init__()
#         self.alpha_min = alpha_min
#         self.feat_channels = feat_channels
#
#         # Shared trunk
#         layers = []
#         ch = in_channels
#         for _ in range(num_layers):
#             layers.append(nn.Linear(ch, hidden_channels))
#             layers.append(nn.ReLU(inplace=True))
#             ch = hidden_channels
#         self.trunk = nn.Sequential(*layers)
#
#         # Separate heads
#         self.offset_head = nn.Linear(hidden_channels, 3)      # Δp
#         self.cov_head = nn.Linear(hidden_channels, 6)          # R_cov compact
#         self.opacity_head = nn.Linear(hidden_channels, 1)      # α (pre-sigmoid)
#         self.feat_head = nn.Linear(hidden_channels, feat_channels)  # e_i
#
#         self._init_weights()
#
#     def _init_weights(self):
#         """Initialize offset head to near-zero for stable coarse-to-fine."""
#         nn.init.zeros_(self.offset_head.weight)
#         nn.init.zeros_(self.offset_head.bias)
#         # Initialize opacity bias so sigmoid(bias) ≈ 0.5
#         nn.init.zeros_(self.opacity_head.bias)
#
#     def forward(self, xyz: Tensor, features: Tensor) -> dict:
#         """
#         Args:
#             xyz (Tensor): Point coordinates, shape (N, 3).
#             features (Tensor): Enhanced point features F_exp, shape (N, C_in).
#
#         Returns:
#             dict with keys:
#                 'means3D': (N, 3) - Gaussian centers μ_i = xyz_i + Δp_i
#                 'cov3D_precomp': (N, 6) - Precomputed 3D covariance (compact)
#                 'opacities': (N, 1) - Opacity after sigmoid + threshold
#                 'features': (N, C_feat) - Implicit feature vectors
#                 'mask': (N,) - Boolean mask for valid (non-pruned) Gaussians
#         """
#         x = torch.cat([xyz, features], dim=-1)  # (N, 3+C_in)
#         h = self.trunk(x)  # (N, hidden)
#
#         # Position: coarse-to-fine
#         offset = self.offset_head(h)       # (N, 3)
#         means3D = xyz + offset             # (N, 3)
#
#         # Covariance: predict 6D parameters, construct PD matrix via L·L^T
#         # MLP predicts 6 values interpreted as lower-triangular Cholesky factor:
#         #   L = [[l0, 0, 0],
#         #        [l1, l2, 0],
#         #        [l3, l4, l5]]
#         # Diagonal elements l0, l2, l5 are passed through softplus to stay positive,
#         # ensuring Σ = L·L^T is always positive definite.
#         # The resulting 6D upper-triangle [Σ00, Σ01, Σ02, Σ11, Σ12, Σ22] is stored
#         # as the precomputed covariance for the rasterizer.
#         cov_raw = self.cov_head(h)         # (N, 6)
#         # Cholesky lower-triangular: [l0, l1, l2, l3, l4, l5]
#         l0 = F.softplus(cov_raw[:, 0])     # diagonal, must be positive
#         l1 = cov_raw[:, 1]                 # off-diagonal, free
#         l2 = F.softplus(cov_raw[:, 2])     # diagonal, must be positive
#         l3 = cov_raw[:, 3]                 # off-diagonal, free
#         l4 = cov_raw[:, 4]                 # off-diagonal, free
#         l5 = F.softplus(cov_raw[:, 5])     # diagonal, must be positive
#         # Σ = L·L^T upper triangle: [xx, xy, xz, yy, yz, zz]
#         cov_xx = l0 * l0
#         cov_xy = l0 * l1
#         cov_xz = l0 * l3
#         cov_yy = l1 * l1 + l2 * l2
#         cov_yz = l1 * l3 + l2 * l4
#         cov_zz = l3 * l3 + l4 * l4 + l5 * l5
#         cov_compact = torch.stack(
#             [cov_xx, cov_xy, cov_xz, cov_yy, cov_yz, cov_zz], dim=-1)  # (N, 6)
#
#         # Opacity: sigmoid with min threshold
#         alpha_raw = self.opacity_head(h)   # (N, 1)
#         opacities = torch.sigmoid(alpha_raw)  # (N, 1)
#
#         # Prune Gaussians with negligible contribution
#         mask = (opacities.squeeze(-1) >= self.alpha_min)
#
#         # Implicit features
#         impl_feats = self.feat_head(h)     # (N, C_feat)
#
#         return {
#             'means3D': means3D,
#             'cov3D_precomp': cov_compact,
#             'opacities': opacities,
#             'features': impl_feats,
#             'mask': mask,
#         }
#
#
# def _build_cov3D_from_compact(cov_compact: Tensor) -> Tensor:
#     """Convert compact 6D covariance to full 6D format for diff-gaussian-rasterization.
#
#     The diff-gaussian-rasterization library expects precomputed 3D covariance
#     in the format: [Σ00, Σ01, Σ02, Σ11, Σ12, Σ22] (upper triangle).
#
#     Our compact representation is [xx, xy, xz, yy, yz, zz] which directly
#     maps to the upper triangle format.
#
#     Args:
#         cov_compact (Tensor): (N, 6) compact covariance.
#
#     Returns:
#         Tensor: (N, 6) covariance in rasterizer format.
#     """
#     # Our format: [xx, xy, xz, yy, yz, zz]
#     # Rasterizer format: [Σ00, Σ01, Σ02, Σ11, Σ12, Σ22]
#     # They are the same mapping!
#     return cov_compact
#
#
# class ImplicitConstraintBranch(BaseModule):
#     """Implicit Constraint Branch using 3D Gaussian Splatting.
#
#     This branch takes the explicit-enhanced point features F_exp and
#     the original point coordinates, converts them to 3D Gaussians,
#     renders them to the image plane via differentiable splatting,
#     and produces a feature map F̂_img for consistency loss with
#     the image encoder features F_img.
#
#     Only active during training. During inference, this entire branch
#     is skipped, but its regularization effect persists in the backbone.
#
#     Args:
#         point_feat_channels (int): Dimension of F_exp (point features).
#         image_feat_channels (int): Dimension of image encoder features.
#             Must match the MLP output feature dimension.
#         hidden_channels (int): MLP hidden layer dimension. Default: 128.
#         num_mlp_layers (int): Number of MLP hidden layers. Default: 2.
#         alpha_min (float): Minimum opacity threshold. Default: 0.01.
#         render_scale (float): Downscale factor for rendering resolution.
#             Default: 1.0 (render at image encoder feature map resolution).
#     """
#
#     def __init__(self,
#                  point_feat_channels: int,
#                  image_feat_channels: int = 128,
#                  hidden_channels: int = 128,
#                  num_mlp_layers: int = 2,
#                  alpha_min: float = 0.01,
#                  render_scale: float = 1.0) -> None:
#         super().__init__()
#
#         assert HAS_DIFF_GAUSSIAN_RASTERIZATION, \
#             ("diff_gaussian_rasterization is not installed. "
#              "Please install it from gaussiancar/ops/diff-gaussian-rasterization/ "
#              "or use the original 3DGS version. "
#              "IMPORTANT: GaussianCaR's version uses BEV orthographic projection. "
#              "For perspective projection to image plane as described in our method, "
#              "you must either (a) use the original 3DGS diff-gaussian-rasterization "
#              "(https://github.com/graphdeco-inria/diff-gaussian-rasterization) "
#              "with NUM_CHANNELS set to 128, or (b) modify GaussianCaR's version "
#              "to restore perspective projection (un-comment the original "
#              "transformPoint4x4/computeCov2D and comment out compute3DtoBEV/"
#              "computeCov2DBEV in forward.cu/backward.cu).")
#
#         self.image_feat_channels = image_feat_channels
#         self.render_scale = render_scale
#
#         # MLP: (3 + point_feat_channels) → Gaussian params
#         self.mlp = PointsToGaussiansMLP(
#             in_channels=3 + point_feat_channels,
#             hidden_channels=hidden_channels,
#             feat_channels=image_feat_channels,
#             num_layers=num_mlp_layers,
#             alpha_min=alpha_min,
#         )
#
#         # Gaussian rasterizer (from diff-gaussian-rasterization)
#         self.rasterizer = GaussianRasterizer()
#
#     def forward(self,
#                 xyz: Tensor,
#                 point_feats: Tensor,
#                 viewmatrix: Tensor,
#                 projmatrix: Tensor,
#                 image_height: int,
#                 image_width: int,
#                 ) -> Tensor:
#         """Render implicit feature map from point cloud via Gaussian Splatting.
#
#         Args:
#             xyz (Tensor): Point coordinates, shape (N, 3).
#             point_feats (Tensor): Enhanced point features F_exp, shape (N, C).
#             viewmatrix (Tensor): Camera extrinsic (world-to-camera), shape (4, 4).
#             projmatrix (Tensor): Full projection matrix (world-to-pixel), shape (4, 4).
#             image_height (int): Target rendering height (at feature map scale).
#             image_width (int): Target rendering width (at feature map scale).
#
#         Returns:
#             Tensor: Rendered implicit feature map F̂_img, shape (C_feat, H, W).
#         """
#         # Step 1: MLP predicts Gaussian parameters
#         gaussian_params = self.mlp(xyz, point_feats)
#
#         means3D = gaussian_params['means3D']        # (N, 3)
#         cov3D = gaussian_params['cov3D_precomp']    # (N, 6)
#         opacities = gaussian_params['opacities']    # (N, 1)
#         impl_feats = gaussian_params['features']    # (N, C_feat)
#         mask = gaussian_params['mask']              # (N,)
#
#         # Apply opacity mask - prune low-opacity Gaussians
#         if mask.sum() < means3D.shape[0]:
#             means3D = means3D[mask]
#             cov3D = cov3D[mask]
#             opacities = opacities[mask]
#             impl_feats = impl_feats[mask]
#
#         N = means3D.shape[0]
#         if N == 0:
#             # No valid Gaussians, return zero feature map
#             return torch.zeros(
#                 self.image_feat_channels, image_height, image_width,
#                 device=xyz.device, dtype=xyz.dtype)
#
#         # Step 2: Build covariance in rasterizer format
#         cov3D_precomp = _build_cov3D_from_compact(cov3D)  # (N, 6)
#
#         # Step 3: Set up rasterization settings
#         # Background color (zeros for feature channels)
#         bg_color = torch.zeros(
#             self.image_feat_channels, device=xyz.device, dtype=torch.float32)
#
#         raster_settings = GaussianRasterizationSettings(
#             image_height=image_height,
#             image_width=image_width,
#             tanfovx=1.0,   # Not used in our modified rasterizer
#             tanfovy=1.0,   # Not used in our modified rasterizer
#             bg=bg_color,
#             scale_modifier=1.0,
#             viewmatrix=viewmatrix.float().contiguous(),
#             projmatrix=projmatrix.float().contiguous(),
#             sh_degree=0,
#             campos=torch.zeros(3, device=xyz.device),  # Not used
#             prefiltered=False,
#             debug=False,
#         )
#         self.rasterizer.set_raster_settings(raster_settings)
#
#         # Step 4: Rasterize
#         # means2D: screen-space positions (will be computed internally but
#         # we need to provide a tensor for gradient tracking)
#         means2D = torch.zeros(N, 2, device=xyz.device, dtype=torch.float32,
#                               requires_grad=True)
#
#         # Use precomputed colors (our implicit features) instead of SH
#         rendered_feat, _ = self.rasterizer(
#             means3D=means3D.float().contiguous(),
#             means2D=means2D,
#             opacities=opacities.float().contiguous(),
#             colors_precomp=impl_feats.float().contiguous(),
#             cov3D_precomp=cov3D_precomp.float().contiguous(),
#         )  # (C_feat, H, W)
#
#         return rendered_feat
#
#
# class ImplicitConstraintLoss(nn.Module):
#     """Implicit constraint loss: L1 distance between rendered and image features.
#
#     L_imp = ||F̂_img - F_img||_1
#
#     where F̂_img is the Gaussian-splatted feature map from point cloud,
#     and F_img is the image encoder intermediate feature map.
#
#     Args:
#         loss_weight (float): Weight λ_imp for the implicit loss. Default: 1.0.
#     """
#
#     def __init__(self, loss_weight: float = 1.0) -> None:
#         super().__init__()
#         self.loss_weight = loss_weight
#
#     def forward(self, rendered_feat: Tensor, image_feat: Tensor) -> Tensor:
#         """
#         Args:
#             rendered_feat (Tensor): F̂_img from Gaussian splatting, (C, H, W) or (B, C, H, W).
#             image_feat (Tensor): F_img from image encoder, same shape.
#
#         Returns:
#             Tensor: Scalar loss value.
#         """
#         # Ensure same spatial resolution
#         if rendered_feat.shape[-2:] != image_feat.shape[-2:]:
#             rendered_feat = F.interpolate(
#                 rendered_feat.unsqueeze(0) if rendered_feat.dim() == 3 else rendered_feat,
#                 size=image_feat.shape[-2:],
#                 mode='bilinear',
#                 align_corners=False,
#             )
#             if rendered_feat.dim() == 4 and image_feat.dim() == 3:
#                 rendered_feat = rendered_feat.squeeze(0)
#
#         loss = F.l1_loss(rendered_feat, image_feat)
#         return self.loss_weight * loss
"""
Implicit Constraint Branch based on 3D Gaussian Splatting.

This module implements the implicit constraint branch that:
1. Maps each point's coordinates + explicit-enhanced features to 3D Gaussian parameters
   via a lightweight MLP (Points-to-Gaussians).
2. Performs differentiable Gaussian Splatting to render an implicit feature map.
3. The rendered feature map is supervised against image encoder features (L_imp).

The branch is ONLY active during training. At inference, it is removed,
but the regularization it imposed on the backbone persists in the learned weights.

Key components:
- PointsToGaussiansMLP: Predicts (μ_offset, Σ_compact, α, e) per point.
- ImplicitConstraintBranch: Orchestrates MLP + differentiable rendering.

References:
- GaussianCaR (Montiel-Marín et al., 2025): Points-to-Gaussians module,
  coarse-to-fine positioning, compact covariance representation.
- 3D Gaussian Splatting (Kerbl et al., 2023): Differentiable rasterization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from torch import Tensor
from typing import Optional, Tuple

try:
    from diff_gaussian_rasterization import (
        GaussianRasterizationSettings,
        GaussianRasterizer,
    )
    HAS_DIFF_GAUSSIAN_RASTERIZATION = True
except ImportError:
    HAS_DIFF_GAUSSIAN_RASTERIZATION = False


class PointsToGaussiansMLP(BaseModule):
    """MLP that maps point coordinates + features to 3D Gaussian parameters.

    Input:  concat(xyz_i, F_exp_i)  →  shape (N, 3 + C_in)
    Output: (offset_3d, cov_compact_6d, opacity_1d, implicit_feat)

    Following GaussianCaR's Points-to-Gaussians design:
    - Position: coarse-to-fine, μ_i = xyz_i + Δp_i (MLP predicts offset only)
    - Covariance: compact 6D representation R_cov ∈ R^6 = [xx,xy,xz,yy,yz,zz],
      eigenvalues enforced positive via softplus → size & orientation
    - Opacity: sigmoid → [0, 1], with α_min = 0.01 threshold
    - Implicit feature: e_i ∈ R^{C_feat}, aligned with image encoder feature dim

    Args:
        in_channels (int): Input feature dimension (3 + C_exp).
        hidden_channels (int): Hidden layer dimension. Default: 128.
        feat_channels (int): Output implicit feature dimension (must match
            image encoder intermediate feature dim). Default: 128.
        num_layers (int): Number of hidden layers. Default: 2.
        alpha_min (float): Minimum opacity threshold. Default: 0.01.
    """

    def __init__(self,
                 in_channels: int,
                 hidden_channels: int = 128,
                 feat_channels: int = 128,
                 num_layers: int = 2,
                 alpha_min: float = 0.01) -> None:
        super().__init__()
        self.alpha_min = alpha_min
        self.feat_channels = feat_channels

        # Shared trunk
        layers = []
        ch = in_channels
        for _ in range(num_layers):
            layers.append(nn.Linear(ch, hidden_channels))
            layers.append(nn.ReLU(inplace=True))
            ch = hidden_channels
        self.trunk = nn.Sequential(*layers)

        # Separate heads
        self.offset_head = nn.Linear(hidden_channels, 3)      # Δp
        self.cov_head = nn.Linear(hidden_channels, 6)          # R_cov compact
        self.opacity_head = nn.Linear(hidden_channels, 1)      # α (pre-sigmoid)
        self.feat_head = nn.Linear(hidden_channels, feat_channels)  # e_i

        self._init_weights()

    def _init_weights(self):
        """Initialize offset head to near-zero for stable coarse-to-fine."""
        nn.init.zeros_(self.offset_head.weight)
        nn.init.zeros_(self.offset_head.bias)
        # Initialize opacity bias so sigmoid(bias) ≈ 0.5
        nn.init.zeros_(self.opacity_head.bias)

    def forward(self, xyz: Tensor, features: Tensor) -> dict:
        """
        Args:
            xyz (Tensor): Point coordinates, shape (N, 3).
            features (Tensor): Enhanced point features F_exp, shape (N, C_in).

        Returns:
            dict with keys:
                'means3D': (N, 3) - Gaussian centers μ_i = xyz_i + Δp_i
                'cov3D_precomp': (N, 6) - Precomputed 3D covariance (compact)
                'opacities': (N, 1) - Opacity after sigmoid + threshold
                'features': (N, C_feat) - Implicit feature vectors
                'mask': (N,) - Boolean mask for valid (non-pruned) Gaussians
        """
        x = torch.cat([xyz, features], dim=-1)  # (N, 3+C_in)
        h = self.trunk(x)  # (N, hidden)

        # Position: coarse-to-fine
        offset = self.offset_head(h)       # (N, 3)
        means3D = xyz + offset             # (N, 3)

        # Covariance: predict 6D parameters, construct PD matrix via L·L^T
        # MLP predicts 6 values interpreted as lower-triangular Cholesky factor:
        #   L = [[l0, 0, 0],
        #        [l1, l2, 0],
        #        [l3, l4, l5]]
        # Diagonal elements l0, l2, l5 are passed through softplus to stay positive,
        # ensuring Σ = L·L^T is always positive definite.
        # The resulting 6D upper-triangle [Σ00, Σ01, Σ02, Σ11, Σ12, Σ22] is stored
        # as the precomputed covariance for the rasterizer.
        cov_raw = self.cov_head(h)         # (N, 6)
        # Cholesky lower-triangular: [l0, l1, l2, l3, l4, l5]
        l0 = F.softplus(cov_raw[:, 0])     # diagonal, must be positive
        l1 = cov_raw[:, 1]                 # off-diagonal, free
        l2 = F.softplus(cov_raw[:, 2])     # diagonal, must be positive
        l3 = cov_raw[:, 3]                 # off-diagonal, free
        l4 = cov_raw[:, 4]                 # off-diagonal, free
        l5 = F.softplus(cov_raw[:, 5])     # diagonal, must be positive
        # Σ = L·L^T upper triangle: [xx, xy, xz, yy, yz, zz]
        cov_xx = l0 * l0
        cov_xy = l0 * l1
        cov_xz = l0 * l3
        cov_yy = l1 * l1 + l2 * l2
        cov_yz = l1 * l3 + l2 * l4
        cov_zz = l3 * l3 + l4 * l4 + l5 * l5
        cov_compact = torch.stack(
            [cov_xx, cov_xy, cov_xz, cov_yy, cov_yz, cov_zz], dim=-1)  # (N, 6)

        # Opacity: sigmoid with min threshold
        alpha_raw = self.opacity_head(h)   # (N, 1)
        opacities = torch.sigmoid(alpha_raw)  # (N, 1)

        # Prune Gaussians with negligible contribution
        mask = (opacities.squeeze(-1) >= self.alpha_min)

        # Implicit features
        impl_feats = self.feat_head(h)     # (N, C_feat)

        return {
            'means3D': means3D,
            'cov3D_precomp': cov_compact,
            'opacities': opacities,
            'features': impl_feats,
            'mask': mask,
        }


def _build_cov3D_from_compact(cov_compact: Tensor) -> Tensor:
    """Convert compact 6D covariance to full 6D format for diff-gaussian-rasterization.

    The diff-gaussian-rasterization library expects precomputed 3D covariance
    in the format: [Σ00, Σ01, Σ02, Σ11, Σ12, Σ22] (upper triangle).

    Our compact representation is [xx, xy, xz, yy, yz, zz] which directly
    maps to the upper triangle format.

    Args:
        cov_compact (Tensor): (N, 6) compact covariance.

    Returns:
        Tensor: (N, 6) covariance in rasterizer format.
    """
    # Our format: [xx, xy, xz, yy, yz, zz]
    # Rasterizer format: [Σ00, Σ01, Σ02, Σ11, Σ12, Σ22]
    # They are the same mapping!
    return cov_compact


class ImplicitConstraintBranch(BaseModule):
    """Implicit Constraint Branch using 3D Gaussian Splatting.

    This branch takes the explicit-enhanced point features F_exp and
    the original point coordinates, converts them to 3D Gaussians,
    renders them to the image plane via differentiable splatting,
    and produces a feature map F̂_img for consistency loss with
    the image encoder features F_img.

    Only active during training. During inference, this entire branch
    is skipped, but its regularization effect persists in the backbone.

    **Multi-pass rendering**: Since the standard diff-gaussian-rasterization
    compiles with NUM_CHANNELS=3 (larger values cause CUDA shared memory /
    register overflow), we render the full C-dimensional feature map by
    splitting the feature channels into groups of 3, performing one
    rasterization pass per group, and concatenating the results.
    This keeps the CUDA kernel unchanged while supporting arbitrary
    feature dimensions.

    Args:
        point_feat_channels (int): Dimension of F_exp (point features).
        image_feat_channels (int): Dimension of image encoder features.
            Must match the MLP output feature dimension. Default: 128.
        hidden_channels (int): MLP hidden layer dimension. Default: 128.
        num_mlp_layers (int): Number of MLP hidden layers. Default: 2.
        alpha_min (float): Minimum opacity threshold. Default: 0.01.
        channels_per_pass (int): Number of feature channels rendered per
            rasterization pass. Must equal NUM_CHANNELS in config.h (default 3).
    """

    def __init__(self,
                 point_feat_channels: int,
                 image_feat_channels: int = 128,
                 hidden_channels: int = 128,
                 num_mlp_layers: int = 2,
                 alpha_min: float = 0.01,
                 channels_per_pass: int = 3) -> None:
        super().__init__()

        assert HAS_DIFF_GAUSSIAN_RASTERIZATION, \
            ("diff_gaussian_rasterization is not installed. "
             "Please install it with NUM_CHANNELS=3 (default). "
             "We use multi-pass rendering to support arbitrary "
             "feature dimensions without modifying the CUDA kernel.")

        self.image_feat_channels = image_feat_channels
        self.channels_per_pass = channels_per_pass

        # MLP: (3 + point_feat_channels) → Gaussian params
        self.mlp = PointsToGaussiansMLP(
            in_channels=3 + point_feat_channels,
            hidden_channels=hidden_channels,
            feat_channels=image_feat_channels,
            num_layers=num_mlp_layers,
            alpha_min=alpha_min,
        )

        # Gaussian rasterizer (from diff-gaussian-rasterization)
        self.rasterizer = GaussianRasterizer()

    def _render_one_pass(self,
                         means3D: Tensor,
                         means2D: Tensor,
                         opacities: Tensor,
                         feat_slice: Tensor,
                         cov3D_precomp: Tensor,
                         raster_settings) -> Tensor:
        """Render one group of feature channels.

        Args:
            feat_slice: (N, C_pass) feature slice, C_pass = channels_per_pass.

        Returns:
            (C_pass, H, W) rendered feature slice.
        """
        self.rasterizer.set_raster_settings(raster_settings)
        rendered, _ = self.rasterizer(
            means3D=means3D,
            means2D=means2D,
            opacities=opacities,
            colors_precomp=feat_slice,
            cov3D_precomp=cov3D_precomp,
        )
        return rendered  # (C_pass, H, W)

    def forward(self,
                xyz: Tensor,
                point_feats: Tensor,
                viewmatrix: Tensor,
                projmatrix: Tensor,
                image_height: int,
                image_width: int,
                ) -> Tensor:
        """Render implicit feature map via multi-pass Gaussian Splatting.

        Splits the C-dimensional implicit features into groups of
        `channels_per_pass` (matching NUM_CHANNELS in the CUDA kernel),
        renders each group separately, then concatenates the results.

        Args:
            xyz (Tensor): Point coordinates, shape (N, 3).
            point_feats (Tensor): Enhanced point features F_exp, shape (N, C).
            viewmatrix (Tensor): Camera extrinsic (world-to-camera), shape (4, 4).
            projmatrix (Tensor): Full projection matrix, shape (4, 4).
            image_height (int): Target rendering height.
            image_width (int): Target rendering width.

        Returns:
            Tensor: Rendered implicit feature map F̂_img, shape (C_feat, H, W).
        """
        # Step 1: MLP predicts Gaussian parameters
        gaussian_params = self.mlp(xyz, point_feats)

        means3D = gaussian_params['means3D']        # (N, 3)
        cov3D = gaussian_params['cov3D_precomp']    # (N, 6)
        opacities = gaussian_params['opacities']    # (N, 1)
        impl_feats = gaussian_params['features']    # (N, C_feat)
        mask = gaussian_params['mask']              # (N,)

        # Apply opacity mask
        if mask.sum() < means3D.shape[0]:
            means3D = means3D[mask]
            cov3D = cov3D[mask]
            opacities = opacities[mask]
            impl_feats = impl_feats[mask]

        N = means3D.shape[0]
        if N == 0:
            return torch.zeros(
                self.image_feat_channels, image_height, image_width,
                device=xyz.device, dtype=xyz.dtype)

        # Prepare common tensors (float, contiguous, on correct device)
        cov3D_precomp = _build_cov3D_from_compact(cov3D).float().contiguous()
        means3D_f = means3D.float().contiguous()
        opacities_f = opacities.float().contiguous()

        C_total = self.image_feat_channels
        C_pass = self.channels_per_pass

        # Pad feature dimension to be divisible by C_pass
        if C_total % C_pass != 0:
            pad_size = C_pass - (C_total % C_pass)
            impl_feats = torch.cat([
                impl_feats,
                torch.zeros(N, pad_size, device=impl_feats.device,
                            dtype=impl_feats.dtype)
            ], dim=-1)
        else:
            pad_size = 0

        C_padded = C_total + pad_size
        num_passes = C_padded // C_pass

        # Step 2: Multi-pass rendering
        rendered_slices = []

        for p in range(num_passes):
            ch_start = p * C_pass
            ch_end = ch_start + C_pass
            feat_slice = impl_feats[:, ch_start:ch_end].float().contiguous()  # (N, C_pass)

            # Background color for this pass
            bg_color = torch.zeros(C_pass, device=xyz.device, dtype=torch.float32)

            raster_settings = GaussianRasterizationSettings(
                image_height=image_height,
                image_width=image_width,
                tanfovx=1.0,
                tanfovy=1.0,
                bg=bg_color,
                scale_modifier=1.0,
                viewmatrix=viewmatrix.float().contiguous(),
                projmatrix=projmatrix.float().contiguous(),
                sh_degree=0,
                campos=torch.zeros(3, device=xyz.device),
                prefiltered=False,
                debug=False,
            )

            # means2D needs fresh gradient tracking per pass
            means2D = torch.zeros(N, 2, device=xyz.device,
                                  dtype=torch.float32, requires_grad=True)

            rendered = self._render_one_pass(
                means3D_f, means2D, opacities_f,
                feat_slice, cov3D_precomp, raster_settings,
            )  # (C_pass, H, W)

            rendered_slices.append(rendered)

        # Step 3: Concatenate all passes
        rendered_full = torch.cat(rendered_slices, dim=0)  # (C_padded, H, W)

        # Remove padding channels
        if pad_size > 0:
            rendered_full = rendered_full[:C_total]

        return rendered_full  # (C_feat, H, W)


class ImplicitConstraintLoss(nn.Module):
    """Implicit constraint loss: L1 distance between rendered and image features.

    L_imp = ||F̂_img - F_img||_1

    where F̂_img is the Gaussian-splatted feature map from point cloud,
    and F_img is the image encoder intermediate feature map.

    Args:
        loss_weight (float): Weight λ_imp for the implicit loss. Default: 1.0.
    """

    def __init__(self, loss_weight: float = 1.0) -> None:
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, rendered_feat: Tensor, image_feat: Tensor) -> Tensor:
        """
        Args:
            rendered_feat (Tensor): F̂_img from Gaussian splatting, (C, H, W) or (B, C, H, W).
            image_feat (Tensor): F_img from image encoder, same shape.

        Returns:
            Tensor: Scalar loss value.
        """
        # Ensure same spatial resolution
        if rendered_feat.shape[-2:] != image_feat.shape[-2:]:
            rendered_feat = F.interpolate(
                rendered_feat.unsqueeze(0) if rendered_feat.dim() == 3 else rendered_feat,
                size=image_feat.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )
            if rendered_feat.dim() == 4 and image_feat.dim() == 3:
                rendered_feat = rendered_feat.squeeze(0)

        loss = F.l1_loss(rendered_feat, image_feat)
        return self.loss_weight * loss