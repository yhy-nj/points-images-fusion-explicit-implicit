"""
Modified Data Preprocessor for Explicit Constraint Branch.

Extends FrustumRangePreprocessor to additionally:
1. Load and preprocess camera images
2. Compute 3D-to-2D projection coordinates using camera calibration
3. Store projection info in voxel_dict for use by the explicit constraint branch
"""

from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import SampleList
from mmengine.model import BaseDataPreprocessor
from torch import Tensor


@MODELS.register_module()
class ExplicitConstraintPreprocessor(BaseDataPreprocessor):
    """Preprocessor that handles both frustum region grouping and image projection.

    In addition to the frustum region group computation from FRNet,
    this preprocessor:
    1. Processes camera images (normalization, resizing)
    2. Computes 3D-to-2D projection of point cloud onto camera image plane
       using extrinsic and intrinsic camera parameters
    3. Passes projection coordinates to the model for explicit constraint fusion

    Args:
        H (int): Height of the 2D frustum representation.
        W (int): Width of the 2D frustum representation.
        fov_up (float): Upward field of view in degrees.
        fov_down (float): Downward field of view in degrees.
        ignore_index (int): Label index to ignore.
        image_size (tuple): Target image size (H, W) for the image backbone.
        mean (list): Image normalization mean.
        std (list): Image normalization std.
        image_stride (int): Output stride of image backbone (for coordinate scaling).
        non_blocking (bool): Whether to use non-blocking data transfer.
    """

    def __init__(self,
                 H: int,
                 W: int,
                 fov_up: float,
                 fov_down: float,
                 ignore_index: int,
                 image_size: tuple = (370, 1226),
                 mean: list = [123.675, 116.28, 103.53],
                 std: list = [58.395, 57.12, 57.375],
                 image_stride: int = 8,
                 non_blocking: bool = False) -> None:
        super().__init__(non_blocking=non_blocking)
        self.H = H
        self.W = W
        self.fov_up = fov_up / 180 * np.pi
        self.fov_down = fov_down / 180 * np.pi
        self.fov = abs(self.fov_down) + abs(self.fov_up)
        self.ignore_index = ignore_index
        self.image_size = image_size
        self.image_stride = image_stride

        # Image normalization parameters
        self.register_buffer(
            'img_mean',
            torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer(
            'img_std',
            torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, data: dict, training: bool = False) -> dict:
        """Process both point cloud and images.

        Args:
            data (dict): Contains 'inputs' and 'data_samples'.
                inputs['points']: list of point cloud tensors
                inputs['imgs']: list of camera images (optional)
                data_samples[i].metainfo may contain:
                    'lidar2img': (4, 4) lidar to image transformation matrix
                    'cam2img': (3, 3) or (4, 4) camera intrinsic matrix
                    'lidar2cam': (4, 4) lidar to camera extrinsic matrix

        Returns:
            dict: Processed data with voxel_dict containing projection info.
        """
        data = self.cast_data(data)
        data.setdefault('data_samples', None)

        inputs, data_samples = data['inputs'], data['data_samples']
        batch_inputs = dict()

        assert 'points' in inputs
        batch_inputs['points'] = inputs['points']

        # Step 1: Frustum region grouping (same as original FRNet)
        voxel_dict = self.frustum_region_group(inputs['points'], data_samples)

        # Step 2: Process images and compute projections
        if 'imgs' in inputs and inputs['imgs'] is not None:
            images, proj_coords = self.process_images_and_project(
                inputs['points'], inputs['imgs'], data_samples)
            voxel_dict['images'] = images  # (B, 3, H_img, W_img)
            voxel_dict['proj_coords'] = proj_coords  # (N_total, 3) [batch, y, x]
            voxel_dict['has_images'] = True
        else:
            voxel_dict['has_images'] = False

        batch_inputs['voxels'] = voxel_dict
        return {'inputs': batch_inputs, 'data_samples': data_samples}

    def process_images_and_project(self,
                                   points: List[Tensor],
                                   imgs: List[Tensor],
                                   data_samples: SampleList):
        """Process camera images and compute 3D-to-2D projections.

        Args:
            points: List of (N_i, 4+) point cloud tensors [x, y, z, intensity, ...]
            imgs: List of (3, H, W) or (H, W, 3) camera image tensors
            data_samples: Contains calibration info in metainfo

        Returns:
            images: (B, 3, H_target, W_target) preprocessed images
            proj_coords: (N_total, 3) [batch_idx, proj_y_in_feat, proj_x_in_feat]
        """
        batch_size = len(points)
        processed_imgs = []
        all_proj_coords = []

        for i in range(batch_size):
            # Process image
            img = imgs[i]
            if img.dim() == 3 and img.shape[-1] == 3:
                img = img.permute(2, 0, 1)  # HWC -> CHW
            img = img.float()

            # Resize to target size
            img = F.interpolate(
                img.unsqueeze(0),
                size=self.image_size,
                mode='bilinear',
                align_corners=False).squeeze(0)  # (3, H_target, W_target)
            processed_imgs.append(img)

            # Compute 3D-to-2D projection
            pts = points[i][:, :3]  # (N_i, 3) xyz

            # Get calibration matrices from metainfo
            meta = data_samples[i].metainfo
            proj_coords_i = self._project_points_to_image(
                pts, meta, i, self.image_size)
            all_proj_coords.append(proj_coords_i)

        # Stack images
        images = torch.stack(processed_imgs, dim=0)  # (B, 3, H, W)

        # Normalize images
        images = (images - self.img_mean) / self.img_std

        # Concatenate projection coordinates
        proj_coords = torch.cat(all_proj_coords, dim=0)  # (N_total, 3)

        # Scale projection coordinates to feature map resolution
        proj_coords[:, 1] = proj_coords[:, 1] / self.image_stride
        proj_coords[:, 2] = proj_coords[:, 2] / self.image_stride

        return images, proj_coords

    def _project_points_to_image(self,
                                 points_3d: Tensor,
                                 meta: dict,
                                 batch_idx: int,
                                 image_size: tuple) -> Tensor:
        """Project 3D points to 2D image coordinates.

        Supports multiple calibration formats:
        1. lidar2img: direct (4, 4) projection matrix
        2. lidar2cam + cam2img: separate extrinsic and intrinsic

        Args:
            points_3d: (N, 3) 3D point coordinates
            meta: metainfo dict with calibration parameters
            batch_idx: batch index
            image_size: (H, W) target image size

        Returns:
            proj_coords: (N, 3) [batch_idx, proj_y, proj_x]
                Points outside image FOV get coordinates clamped to image border.
        """
        N = points_3d.shape[0]
        device = points_3d.device
        H_img, W_img = image_size

        # Homogeneous coordinates
        ones = torch.ones(N, 1, device=device, dtype=points_3d.dtype)
        pts_homo = torch.cat([points_3d, ones], dim=-1)  # (N, 4)

        if 'lidar2img' in meta:
            # Direct projection matrix
            lidar2img = torch.tensor(
                meta['lidar2img'], device=device, dtype=points_3d.dtype)
            if lidar2img.shape == (4, 4):
                pts_img = pts_homo @ lidar2img.T  # (N, 4)
            elif lidar2img.shape == (3, 4):
                pts_img = pts_homo @ lidar2img.T  # (N, 3)
            else:
                # Try reshape
                pts_img = pts_homo @ lidar2img[:4, :4].T

            # Perspective division
            depth = pts_img[:, 2].clamp(min=1e-5)
            proj_x = pts_img[:, 0] / depth
            proj_y = pts_img[:, 1] / depth

        elif 'lidar2cam' in meta and 'cam2img' in meta:
            lidar2cam = torch.tensor(
                meta['lidar2cam'], device=device, dtype=points_3d.dtype)
            cam2img = torch.tensor(
                meta['cam2img'], device=device, dtype=points_3d.dtype)

            # Transform to camera frame
            pts_cam = pts_homo @ lidar2cam.T  # (N, 4)

            # Project to image
            if cam2img.shape == (3, 3):
                pts_img = pts_cam[:, :3] @ cam2img.T  # (N, 3)
            elif cam2img.shape == (3, 4):
                pts_img = pts_cam @ cam2img.T  # (N, 3)
            elif cam2img.shape == (4, 4):
                pts_img = pts_cam @ cam2img.T  # (N, 4)
                pts_img = pts_img[:, :3]
            else:
                pts_img = pts_cam[:, :3] @ cam2img[:3, :3].T

            depth = pts_img[:, 2].clamp(min=1e-5)
            proj_x = pts_img[:, 0] / depth
            proj_y = pts_img[:, 1] / depth

        else:
            # Fallback: use spherical projection (same as FRNet range projection)
            # This is used when camera calibration is not available
            depth = torch.norm(points_3d, dim=-1)
            yaw = -torch.atan2(points_3d[:, 1], points_3d[:, 0])
            pitch = torch.arcsin(points_3d[:, 2] / depth.clamp(min=1e-5))

            proj_x = 0.5 * (yaw / np.pi + 1.0) * W_img
            proj_y = (1.0 - (pitch + abs(self.fov_down)) / self.fov) * H_img

        # Handle original image size scaling if needed
        if 'ori_shape' in meta and 'img_shape' in meta:
            ori_h, ori_w = meta['ori_shape'][:2]
            # Scale to target image_size
            proj_x = proj_x * (W_img / ori_w)
            proj_y = proj_y * (H_img / ori_h)

        # Clamp to valid range
        proj_x = proj_x.clamp(0, W_img - 1)
        proj_y = proj_y.clamp(0, H_img - 1)

        # Create output with batch index
        batch_indices = torch.full((N, 1), batch_idx,
                                   device=device, dtype=proj_x.dtype)
        proj_coords = torch.cat([batch_indices, proj_y.unsqueeze(-1),
                                 proj_x.unsqueeze(-1)], dim=-1)  # (N, 3)

        return proj_coords

    @torch.no_grad()
    def frustum_region_group(self, points: List[Tensor],
                             data_samples: SampleList) -> dict:
        """Calculate frustum region of each point (same as original FRNet).

        Args:
            points: Point cloud in one data batch.

        Returns:
            dict: Frustum region information.
        """
        voxel_dict = dict()
        coors = []
        voxels = []

        for i, res in enumerate(points):
            depth = torch.linalg.norm(res[:, :3], 2, dim=1)
            yaw = -torch.atan2(res[:, 1], res[:, 0])
            pitch = torch.arcsin(res[:, 2] / depth)

            coors_x = 0.5 * (yaw / np.pi + 1.0)
            coors_y = 1.0 - (pitch + abs(self.fov_down)) / self.fov

            coors_x *= self.W
            coors_y *= self.H

            coors_x = torch.floor(coors_x)
            coors_x = torch.clamp(coors_x, min=0, max=self.W - 1).type(torch.int64)

            coors_y = torch.floor(coors_y)
            coors_y = torch.clamp(coors_y, min=0, max=self.H - 1).type(torch.int64)

            res_coors = torch.stack([coors_y, coors_x], dim=1)
            res_coors = F.pad(res_coors, (1, 0), mode='constant', value=i)
            coors.append(res_coors)
            voxels.append(res)

            if 'pts_semantic_mask' in data_samples[i].gt_pts_seg:
                import torch_scatter
                pts_semantic_mask = data_samples[i].gt_pts_seg.pts_semantic_mask
                seg_label = torch.ones(
                    (self.H, self.W),
                    dtype=torch.long,
                    device=pts_semantic_mask.device) * self.ignore_index
                res_voxel_coors, inverse_map = torch.unique(
                    res_coors, return_inverse=True, dim=0)
                voxel_semantic_mask = torch_scatter.scatter_mean(
                    F.one_hot(pts_semantic_mask).float(), inverse_map, dim=0)
                voxel_semantic_mask = torch.argmax(voxel_semantic_mask, dim=-1)
                seg_label[res_voxel_coors[:, 1],
                          res_voxel_coors[:, 2]] = voxel_semantic_mask
                data_samples[i].gt_pts_seg.semantic_seg = seg_label

        voxels = torch.cat(voxels, dim=0)
        coors = torch.cat(coors, dim=0)
        voxel_dict['voxels'] = voxels
        voxel_dict['coors'] = coors

        return voxel_dict
