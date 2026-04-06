"""
FRNet Segmentor with Explicit + Implicit Constraint Branches.

This extends FRNetExplicit to add the implicit constraint loss:

    L = L_seg + λ_exp · L_exp + λ_imp · L_imp

where:
    L_seg  = FRNet main segmentation loss (point-level + frustum-level)
    L_exp  = Contrastive alignment loss (explicit cross-modal constraint)
    L_imp  = ||F̂_img - F_img||_1 (implicit Gaussian splatting constraint)

The implicit branch is ONLY active during training. At inference:
    - The implicit branch and its MLP are completely removed
    - Only the backbone + explicit branch run (same as FRNetExplicit)
    - The regularization from implicit training persists in backbone weights

This is the key design advantage: train-time-only regularization that
improves feature spatial continuity and cross-modal consistency without
adding any inference overhead.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import OptSampleList, SampleList
from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig
from torch import Tensor

from .frnet_explicit import FRNetExplicit
from ..backbones.implicit_constraint import ImplicitConstraintLoss


@MODELS.register_module()
class FRNetExplicitImplicit(FRNetExplicit):
    """FRNet with Explicit + Implicit Constraint Branches.

    Extends FRNetExplicit by adding the implicit constraint loss
    from 3D Gaussian Splatting rendered feature maps.

    Total loss: L = L_seg + λ_exp · L_exp + λ_imp · L_imp

    Args:
        voxel_encoder (dict): Config for voxel encoder.
        backbone (dict): Config for backbone (FRNetExplicitImplicitBackbone).
        decode_head (dict): Config for decode head.
        neck (dict, optional): Config for neck.
        auxiliary_head (dict, optional): Config for auxiliary head.
        implicit_loss_weight (float): Weight λ_imp for implicit loss.
            Default: 1.0.
        train_cfg (dict, optional): Training config.
        test_cfg (dict, optional): Testing config.
        data_preprocessor (dict, optional): Data preprocessor config.
        init_cfg (dict, optional): Initialization config.
    """

    def __init__(self,
                 voxel_encoder: ConfigType,
                 backbone: ConfigType,
                 decode_head: ConfigType,
                 neck: OptConfigType = None,
                 auxiliary_head: OptMultiConfig = None,
                 implicit_loss_weight: float = 1.0,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:

        super().__init__(
            voxel_encoder=voxel_encoder,
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg,
        )

        # Implicit constraint loss
        self.implicit_loss = ImplicitConstraintLoss(
            loss_weight=implicit_loss_weight)

    def loss(self, batch_inputs_dict: dict,
             batch_data_samples: SampleList) -> Dict[str, Tensor]:
        """Calculate losses including implicit constraint loss.

        Total loss = L_seg + λ_exp · L_exp + λ_imp · L_imp

        where:
            L_seg = L_point + L_frustum (from decode head + auxiliary heads)
            L_exp = Contrastive alignment loss (from explicit branch)
            L_imp = ||F̂_img - F_img||_1 (from implicit branch)
        """
        # Extract features (runs backbone with explicit + implicit branches)
        voxel_dict = self.extract_feat(batch_inputs_dict)
        losses = dict()

        # ---- Original segmentation losses (L_seg) ----
        loss_decode = self._decode_head_forward_train(
            voxel_dict, batch_data_samples)
        losses.update(loss_decode)

        # Auxiliary head loss (L_frustum)
        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                voxel_dict, batch_data_samples)
            losses.update(loss_aux)

        # ---- Explicit Constraint Loss (L_exp) ----
        if 'z_voxel' in voxel_dict and 'z_image' in voxel_dict:
            gt_semantic_segs = [
                data_sample.gt_pts_seg.pts_semantic_mask
                for data_sample in batch_data_samples
            ]
            semantic_labels = torch.cat(gt_semantic_segs, dim=0)

            loss_contrastive = self.contrastive_loss(
                z_voxel=voxel_dict['z_voxel'],
                z_image=voxel_dict['z_image'],
                semantic_labels=semantic_labels)
            losses['loss_contrastive'] = loss_contrastive

        # ---- Implicit Constraint Loss (L_imp) ----
        if ('rendered_implicit_feat' in voxel_dict
                and 'image_feat_for_implicit' in voxel_dict):
            rendered_feat = voxel_dict['rendered_implicit_feat']  # (B, C, H, W)
            image_feat = voxel_dict['image_feat_for_implicit']    # (B, C, H, W)

            loss_implicit = self.implicit_loss(rendered_feat, image_feat)
            losses['loss_implicit'] = loss_implicit

        return losses

    def predict(self,
                batch_inputs_dict: dict,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Inference - implicit branch is automatically skipped.

        Since self.training is False during inference, the backbone's
        forward method skips the implicit branch entirely.
        """
        return super().predict(batch_inputs_dict, batch_data_samples, rescale)

    def _forward(self,
                 batch_inputs_dict: dict,
                 batch_data_samples: OptSampleList = None) -> dict:
        """Network forward process."""
        voxel_dict = self.extract_feat(batch_inputs_dict)
        return self.decode_head.forward(voxel_dict)
