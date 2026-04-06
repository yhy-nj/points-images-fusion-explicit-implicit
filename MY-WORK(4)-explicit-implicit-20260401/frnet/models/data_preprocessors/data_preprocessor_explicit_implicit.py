"""
Data Preprocessor Modifications for Implicit Constraint.

This file provides the minimal modification needed in
frnet/models/data_preprocessors/data_preprocessor_explicit.py
to pass lidar2img and raw point xyz through the pipeline
for the implicit constraint branch.

Option A: Modify existing FRNetExplicitDataPreprocessor
Option B: Create a new subclass (shown below)
"""

import torch
import torch.nn.functional as F
from typing import List
from torch import Tensor
from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import SampleList

# Import the existing data preprocessor
from .data_preprocessor_explicit import ExplicitConstraintPreprocessor


@MODELS.register_module()
class FRNetExplicitImplicitDataPreprocessor(ExplicitConstraintPreprocessor):
    """Data Preprocessor with additional fields for implicit constraint.

    Extends FRNetExplicitDataPreprocessor by additionally storing:
    - lidar2img matrix in voxel_dict (for 3D→2D projection in Gaussian splatting)
    - raw point xyz coordinates in voxel_dict (for MLP input)

    These fields are needed by the implicit constraint branch during training.
    """

    def forward(self, data: dict, training: bool = False) -> dict:
        """Preprocess data and add implicit-constraint-specific fields.

        Calls parent's forward, then injects lidar2img and points_xyz
        into voxel_dict.
        """
        # Run parent preprocessing (handles images, proj_coords, frustum, etc.)
        result = super().forward(data, training=training)

        inputs = result['inputs']
        data_samples = result.get('data_samples', None)

        if 'voxels' in inputs:
            voxel_dict = inputs['voxels']

            # ---- Store lidar2img for implicit branch ----
            if data_samples is not None and len(data_samples) > 0:
                meta = data_samples[0].metainfo
                if 'lidar2img' in meta:
                    lidar2img = meta['lidar2img']
                    if not isinstance(lidar2img, torch.Tensor):
                        lidar2img = torch.tensor(lidar2img, dtype=torch.float32)
                    # Ensure 4x4
                    if lidar2img.shape == (3, 4):
                        lidar2img_4x4 = torch.eye(4, dtype=torch.float32)
                        lidar2img_4x4[:3, :4] = lidar2img
                        lidar2img = lidar2img_4x4
                    voxel_dict['lidar2img'] = lidar2img

            # ---- Store raw point xyz for implicit branch ----
            # The original points (before voxelization) are stored by parent
            # in batch_inputs['points']. We need to extract xyz from voxels.
            if 'voxels' in voxel_dict:
                # voxel_dict['voxels'] contains per-point features [x,y,z,intensity,...]
                raw_points = voxel_dict['voxels']
                if raw_points.dim() == 2 and raw_points.shape[-1] >= 3:
                    voxel_dict['points_xyz'] = raw_points[:, :3].clone()

        return result
