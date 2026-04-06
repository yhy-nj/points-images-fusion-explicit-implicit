# """
# Modified FRNet Segmentor with Explicit Constraint Branch.
#
# This extends the original FRNet segmentor to support:
# 1. Image backbone for extracting 2D features
# 2. Explicit constraint branch for cross-modal fusion
# 3. Modified data flow to pass images and calibration info through the pipeline
# """
#
# from typing import Dict, Optional
#
# from mmdet3d.models import EncoderDecoder3D
# from mmdet3d.registry import MODELS
# from mmdet3d.structures.det3d_data_sample import OptSampleList, SampleList
# from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig
# from torch import Tensor
#
#
# @MODELS.register_module()
# class FRNetExplicit(EncoderDecoder3D):
#     """FRNet with Explicit Constraint Branch.
#
#     This segmentor extends FRNet by adding an explicit constraint branch
#     that fuses image features with 3D point cloud features for improved
#     semantic segmentation.
#
#     The explicit constraint branch is integrated into the backbone
#     (FRNetExplicitBackbone), so the segmentor mainly handles:
#     1. Building the voxel encoder, backbone (with explicit branch), and head
#     2. Passing image and calibration data through the pipeline
#
#     Args:
#         voxel_encoder (dict): Config for voxel encoder.
#         backbone (dict): Config for backbone (FRNetExplicitBackbone).
#         decode_head (dict): Config for decode head.
#         neck (dict, optional): Config for neck.
#         auxiliary_head (dict, optional): Config for auxiliary head.
#         train_cfg (dict, optional): Training config.
#         test_cfg (dict, optional): Testing config.
#         data_preprocessor (dict, optional): Data preprocessor config.
#         init_cfg (dict, optional): Initialization config.
#     """
#
#     def __init__(self,
#                  voxel_encoder: ConfigType,
#                  backbone: ConfigType,
#                  decode_head: ConfigType,
#                  neck: OptConfigType = None,
#                  auxiliary_head: OptMultiConfig = None,
#                  train_cfg: OptConfigType = None,
#                  test_cfg: OptConfigType = None,
#                  data_preprocessor: OptConfigType = None,
#                  init_cfg: OptMultiConfig = None) -> None:
#         super(FRNetExplicit, self).__init__(
#             backbone=backbone,
#             decode_head=decode_head,
#             neck=neck,
#             auxiliary_head=auxiliary_head,
#             train_cfg=train_cfg,
#             test_cfg=test_cfg,
#             data_preprocessor=data_preprocessor,
#             init_cfg=init_cfg)
#
#         self.voxel_encoder = MODELS.build(voxel_encoder)
#
#     def extract_feat(self, batch_inputs_dict: dict) -> dict:
#         """Extract features from points and images.
#
#         The voxel_dict now contains:
#         - Original FRNet voxel data (voxels, coors, etc.)
#         - images: (B, 3, H, W) camera images (if available)
#         - proj_coords: (N, 3) projection coordinates (if available)
#         - has_images: bool flag
#         """
#         voxel_dict = batch_inputs_dict['voxels'].copy()
#
#         # Step 1: Voxel encoder (same as original)
#         voxel_dict = self.voxel_encoder(voxel_dict)
#
#         # Step 2: Backbone with explicit constraint
#         # The backbone (FRNetExplicitBackbone) handles both:
#         # - Original FRNet feature extraction
#         # - Explicit constraint branch (if images are available)
#         voxel_dict = self.backbone(voxel_dict)
#
#         if self.with_neck:
#             voxel_dict = self.neck(voxel_dict)
#
#         return voxel_dict
#
#     def loss(self, batch_inputs_dict: dict,
#              batch_data_samples: SampleList) -> Dict[str, Tensor]:
#         """Calculate losses.
#
#         Same as original FRNet but passes image info through.
#         """
#         voxel_dict = self.extract_feat(batch_inputs_dict)
#         losses = dict()
#         loss_decode = self._decode_head_forward_train(
#             voxel_dict, batch_data_samples)
#         losses.update(loss_decode)
#
#         if self.with_auxiliary_head:
#             loss_aux = self._auxiliary_head_forward_train(
#                 voxel_dict, batch_data_samples)
#             losses.update(loss_aux)
#         return losses
#
#     def predict(self,
#                 batch_inputs_dict: dict,
#                 batch_data_samples: SampleList,
#                 rescale: bool = True) -> SampleList:
#         """Simple test with single scene."""
#         batch_input_metas = []
#         for data_sample in batch_data_samples:
#             batch_input_metas.append(data_sample.metainfo)
#
#         voxel_dict = self.extract_feat(batch_inputs_dict)
#         seg_logits_list = self.decode_head.predict(
#             voxel_dict, batch_input_metas, self.test_cfg)
#         for i in range(len(seg_logits_list)):
#             seg_logits_list[i] = seg_logits_list[i].transpose(0, 1)
#
#         return self.postprocess_result(seg_logits_list, batch_data_samples)
#
#     def _forward(self,
#                  batch_inputs_dict: dict,
#                  batch_data_samples: OptSampleList = None) -> dict:
#         """Network forward process."""
#         voxel_dict = self.extract_feat(batch_inputs_dict)
#         return self.decode_head.forward(voxel_dict)
"""
Modified FRNet Segmentor with Explicit Constraint Branch.

This extends the original FRNet segmentor to support:
1. Image backbone for extracting 2D features
2. Explicit constraint branch for cross-modal fusion
3. Contrastive alignment loss (Loss_VI) for explicit cross-modal constraint
4. Modified data flow to pass images and calibration info through the pipeline
"""

from typing import Dict, Optional

from mmdet3d.models import EncoderDecoder3D
from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import OptSampleList, SampleList
from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig
from torch import Tensor

import torch

from frnet.models.losses.contrastive_loss import ContrastiveAlignmentLossEfficient


@MODELS.register_module()
class FRNetExplicit(EncoderDecoder3D):
    """FRNet with Explicit Constraint Branch.

    This segmentor extends FRNet by adding:
    1. Explicit constraint branch (in backbone) for cross-modal fusion
    2. Contrastive alignment loss (Loss_VI) that provides the "constraint"
       by enforcing cross-modal feature alignment in a shared embedding space

    Additional Args:
        contrastive_loss (dict, optional): Config for contrastive alignment loss.
    """

    def __init__(self,
                 voxel_encoder: ConfigType,
                 backbone: ConfigType,
                 decode_head: ConfigType,
                 neck: OptConfigType = None,
                 auxiliary_head: OptMultiConfig = None,
                 contrastive_loss: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super(FRNetExplicit, self).__init__(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)

        self.voxel_encoder = MODELS.build(voxel_encoder)

        # Contrastive alignment loss (the "constraint" in explicit constraint)
        if contrastive_loss is not None:
            self.contrastive_loss = MODELS.build(contrastive_loss)
        else:
            self.contrastive_loss = ContrastiveAlignmentLossEfficient(
                temperature=0.07,
                loss_weight=0.1,
                ignore_index=19,
                max_points=4096)

    def extract_feat(self, batch_inputs_dict: dict) -> dict:
        """Extract features from points and images."""
        voxel_dict = batch_inputs_dict['voxels'].copy()
#===========================新增===============================
        if 'lidar2img' in batch_inputs_dict:
            voxel_dict['lidar2img'] = batch_inputs_dict['lidar2img']
#===========================新增===============================

        voxel_dict = self.voxel_encoder(voxel_dict)
        voxel_dict = self.backbone(voxel_dict)
        if self.with_neck:
            voxel_dict = self.neck(voxel_dict)
        return voxel_dict

    def loss(self, batch_inputs_dict: dict,
             batch_data_samples: SampleList) -> Dict[str, Tensor]:
        """Calculate losses including contrastive alignment loss.

        Total loss = L_point + λ_f * L_frustum + λ_c * Loss_VI

        where Loss_VI is the contrastive alignment loss that provides
        the explicit constraint for cross-modal feature alignment.
        """
        voxel_dict = self.extract_feat(batch_inputs_dict)
        losses = dict()

        # Original decode head loss (L_point)
        loss_decode = self._decode_head_forward_train(
            voxel_dict, batch_data_samples)
        losses.update(loss_decode)

        # Auxiliary head loss (L_frustum)
        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                voxel_dict, batch_data_samples)
            losses.update(loss_aux)

        # ---- Contrastive Alignment Loss (Loss_VI) ----
        # This is the core "constraint" that makes this branch "explicit"
        if 'z_voxel' in voxel_dict and 'z_image' in voxel_dict:
            # Get per-point semantic labels
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

        return losses

    def predict(self,
                batch_inputs_dict: dict,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Simple test with single scene."""
        batch_input_metas = []
        for data_sample in batch_data_samples:
            batch_input_metas.append(data_sample.metainfo)

        voxel_dict = self.extract_feat(batch_inputs_dict)
        seg_logits_list = self.decode_head.predict(
            voxel_dict, batch_input_metas, self.test_cfg)
        for i in range(len(seg_logits_list)):
            seg_logits_list[i] = seg_logits_list[i].transpose(0, 1)

        return self.postprocess_result(seg_logits_list, batch_data_samples)

    def _forward(self,
                 batch_inputs_dict: dict,
                 batch_data_samples: OptSampleList = None) -> dict:
        """Network forward process."""
        voxel_dict = self.extract_feat(batch_inputs_dict)
        return self.decode_head.forward(voxel_dict)