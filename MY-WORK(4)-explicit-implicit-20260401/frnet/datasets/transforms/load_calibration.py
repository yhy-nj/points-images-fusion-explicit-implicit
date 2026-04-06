# """LoadCalibration transform for SemanticKITTI dataset.
#
# This module loads camera calibration parameters (calib.txt) from
# SemanticKITTI sequences, enabling LiDAR-camera fusion by providing
# the projection matrices needed to map 3D LiDAR points to 2D image
# pixel coordinates.
#
# Usage:
#     Add to your pipeline config:
#         dict(type='LoadCalibration')
#
#     Place this file in: frnet/datasets/transforms/load_calibration.py
#     Then register it in: frnet/datasets/transforms/__init__.py
# """
#
# import os
# import numpy as np
# from mmcv.transforms import BaseTransform
# from mmdet3d.registry import TRANSFORMS
#
#
# @TRANSFORMS.register_module()
# class LoadCalibration(BaseTransform):
#     """Load calibration parameters from SemanticKITTI calib.txt.
#
#     SemanticKITTI calib.txt contains the following matrices:
#         - P0: Projection matrix for camera 0 (grayscale left)   (3x4)
#         - P1: Projection matrix for camera 1 (grayscale right)  (3x4)
#         - P2: Projection matrix for camera 2 (color left)       (3x4)
#         - P3: Projection matrix for camera 3 (color right)      (3x4)
#         - Tr: LiDAR to camera 0 transformation matrix           (3x4)
#
#     The LiDAR point projection to image pixel coordinates:
#         p_img = P2 @ Tr_homo @ p_lidar_homo
#
#     where:
#         p_lidar_homo = [x, y, z, 1]^T  (homogeneous LiDAR coordinates)
#         Tr_homo = [[Tr], [0, 0, 0, 1]]  (4x4 homogeneous transformation)
#         p_img = [u*z, v*z, z]^T  (homogeneous image coordinates)
#
#     Args:
#         use_camera (str): Which camera projection matrix to use.
#             Options: 'P0', 'P1', 'P2', 'P3'. Default: 'P2' (color left).
#     """
#
#     def __init__(self, use_camera: str = 'P2') -> None:
#         assert use_camera in ('P0', 'P1', 'P2', 'P3'), \
#             f"use_camera must be one of 'P0', 'P1', 'P2', 'P3', got {use_camera}"
#         self.use_camera = use_camera
#
#     def _parse_calib_file(self, calib_path: str) -> dict:
#         """Parse a SemanticKITTI calib.txt file.
#
#         The calib.txt format:
#             P0: 7.070912e+02 0.000000e+00 6.018873e+02 ...
#             P1: 7.070912e+02 0.000000e+00 6.018873e+02 ...
#             P2: 7.070912e+02 0.000000e+00 6.018873e+02 ...
#             P3: 7.070912e+02 0.000000e+00 6.018873e+02 ...
#             Tr: 4.276802e-04 -9.999672e-01 -8.084491e-03 ...
#
#         Args:
#             calib_path (str): Path to calib.txt.
#
#         Returns:
#             dict: Parsed calibration matrices.
#         """
#         calib = {}
#         with open(calib_path, 'r') as f:
#             for line in f.readlines():
#                 line = line.strip()
#                 if not line or ':' not in line:
#                     continue
#                 key, value = line.split(':', 1)
#                 calib[key.strip()] = np.array(
#                     [float(x) for x in value.strip().split()],
#                     dtype=np.float32
#                 )
#         return calib
#
#     def transform(self, input_dict: dict) -> dict:
#         """Load calibration and add to input_dict.
#
#         Args:
#             input_dict (dict): Result dict containing 'lidar_points'.
#
#         Returns:
#             dict: Result dict with added calibration info:
#                 - calib (dict): Contains calibration matrices:
#                     - P0 (np.ndarray): Camera 0 projection matrix (3x4)
#                     - P1 (np.ndarray): Camera 1 projection matrix (3x4)
#                     - P2 (np.ndarray): Camera 2 projection matrix (3x4)
#                     - P3 (np.ndarray): Camera 3 projection matrix (3x4)
#                     - Tr (np.ndarray): LiDAR-to-camera transformation (3x4)
#                     - Tr_homo (np.ndarray): Homogeneous Tr matrix (4x4)
#                     - lidar2img (np.ndarray): Full LiDAR-to-image
#                         projection matrix (3x4), computed as P_cam @ Tr_homo
#         """
#         # ------------------------------------------------------------------
#         # Step 1: Infer calib.txt path from LiDAR point cloud path
#         # ------------------------------------------------------------------
#         # lidar_path example: sequences/02/velodyne/000000.bin
#         # calib.txt location: sequences/02/calib.txt
#         pts_filepath = input_dict['lidar_points']['lidar_path']
#
#         # Handle both relative and absolute paths
#         # Go up two levels: velodyne/000000.bin -> 02/
#         seq_dir = os.path.dirname(os.path.dirname(pts_filepath))
#         calib_path = os.path.join(seq_dir, 'calib.txt')
#
#         # If lidar_path is relative, prepend data_root
#         if not os.path.isabs(calib_path):
#             data_root = input_dict.get('data_root', '')
#             if data_root:
#                 calib_path = os.path.join(data_root, calib_path)
#
#         # If still not found, try using lidar_path directly
#         if not os.path.exists(calib_path):
#             # Try absolute lidar path
#             lidar_path = input_dict['lidar_points'].get('lidar_path', '')
#             if os.path.isabs(lidar_path):
#                 seq_dir = os.path.dirname(os.path.dirname(lidar_path))
#                 calib_path = os.path.join(seq_dir, 'calib.txt')
#
#         assert os.path.exists(calib_path), \
#             f'Calibration file not found: {calib_path}'
#
#         # ------------------------------------------------------------------
#         # Step 2: Parse calib.txt
#         # ------------------------------------------------------------------
#         calib_data = self._parse_calib_file(calib_path)
#
#         # ------------------------------------------------------------------
#         # Step 3: Reshape matrices
#         # ------------------------------------------------------------------
#         # Projection matrices: 3x4
#         P0 = calib_data['P0'].reshape(3, 4)
#         P1 = calib_data['P1'].reshape(3, 4)
#         P2 = calib_data['P2'].reshape(3, 4)
#         P3 = calib_data['P3'].reshape(3, 4)
#
#         # LiDAR to camera 0 transformation: 3x4
#         Tr = calib_data['Tr'].reshape(3, 4)
#
#         # Build 4x4 homogeneous transformation matrix
#         Tr_homo = np.eye(4, dtype=np.float32)
#         Tr_homo[:3, :4] = Tr
#
#         # ------------------------------------------------------------------
#         # Step 4: Compute LiDAR-to-image projection matrix
#         # ------------------------------------------------------------------
#         # P_cam @ Tr_homo -> (3x4) @ (4x4) = (3x4)
#         # This maps homogeneous LiDAR coords [x,y,z,1] to image [u*z, v*z, z]
#         P_cam = locals()[self.use_camera]  # Select camera (default P2)
#         lidar2img = P_cam @ Tr_homo  # (3x4)
#
#         # ------------------------------------------------------------------
#         # Step 5: Store in input_dict
#         # ------------------------------------------------------------------
#         input_dict['calib'] = {
#             'P0': P0,
#             'P1': P1,
#             'P2': P2,
#             'P3': P3,
#             'Tr': Tr,
#             'Tr_homo': Tr_homo,
#             'lidar2img': lidar2img,
#         }
#
#         # Also store as a top-level key for convenience
#         # (some modules may directly look for 'lidar2img')
#         input_dict['lidar2img'] = lidar2img
#
#         return input_dict
#
#     def __repr__(self) -> str:
#         """str: Return a string that describes the module."""
#         repr_str = self.__class__.__name__
#         repr_str += f'(use_camera={self.use_camera})'
#         return repr_str
"""LoadCalibration transform for SemanticKITTI dataset.

This module loads camera calibration parameters (calib.txt) and
infers the corresponding image path from SemanticKITTI sequences,
enabling LiDAR-camera fusion.

Usage:
    Add to your pipeline config (place BEFORE LoadImageFromFile):
        dict(type='LoadCalibration'),
        dict(type='LoadImageFromFile'),

    Place this file in: frnet/datasets/transforms/load_calibration.py
    Then register it in: frnet/datasets/transforms/__init__.py
"""

import os
import numpy as np
from mmcv.transforms import BaseTransform
from mmdet3d.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadCalibration(BaseTransform):
    """Load calibration parameters and infer image path from SemanticKITTI.

    SemanticKITTI data structure:
        sequences/02/
        ├── velodyne/000000.bin      # LiDAR point cloud
        ├── image_2/000000.png       # Color left camera image
        ├── image_3/000000.png       # Color right camera image
        ├── labels/000000.label      # Semantic labels
        └── calib.txt                # Calibration parameters

    calib.txt contains:
        - P0: Projection matrix for camera 0 (grayscale left)   (3x4)
        - P1: Projection matrix for camera 1 (grayscale right)  (3x4)
        - P2: Projection matrix for camera 2 (color left)       (3x4)
        - P3: Projection matrix for camera 3 (color right)      (3x4)
        - Tr: LiDAR to camera 0 transformation matrix           (3x4)

    This transform also infers the image path from the LiDAR path
    and adds it to input_dict so that LoadImageFromFile can work.

    Args:
        use_camera (str): Which camera to use.
            Options: 'P2', 'P3'. Default: 'P2' (color left -> image_2).
        img_suffix (str): Image file suffix. Default: '.png'.
    """

    def __init__(self,
                 use_camera: str = 'P2',
                 img_suffix: str = '.png') -> None:
        assert use_camera in ('P0', 'P1', 'P2', 'P3'), \
            f"use_camera must be 'P0','P1','P2','P3', got {use_camera}"
        self.use_camera = use_camera
        self.img_suffix = img_suffix

        # P2 -> image_2, P3 -> image_3
        self._camera_to_folder = {
            'P0': 'image_0',
            'P1': 'image_1',
            'P2': 'image_2',
            'P3': 'image_3',
        }

    def _parse_calib_file(self, calib_path: str) -> dict:
        """Parse a SemanticKITTI calib.txt file.

        Args:
            calib_path (str): Path to calib.txt.

        Returns:
            dict: Parsed calibration matrices.
        """
        calib = {}
        with open(calib_path, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if not line or ':' not in line:
                    continue
                key, value = line.split(':', 1)
                calib[key.strip()] = np.array(
                    [float(x) for x in value.strip().split()],
                    dtype=np.float32
                )
        return calib

    def _infer_img_path(self, pts_filepath: str, data_root: str = '') -> str:
        """Infer image path from LiDAR point cloud path.

        Example:
            pts_filepath: sequences/02/velodyne/000000.bin
            -> image path: sequences/02/image_2/000000.png

        Args:
            pts_filepath (str): Path to LiDAR point cloud file.
            data_root (str): Data root directory.

        Returns:
            str: Absolute path to corresponding image file.
        """
        frame_name = os.path.splitext(os.path.basename(pts_filepath))[0]
        seq_dir = os.path.dirname(os.path.dirname(pts_filepath))

        img_folder = self._camera_to_folder[self.use_camera]
        img_rel_path = os.path.join(
            seq_dir, img_folder, frame_name + self.img_suffix)

        if not os.path.isabs(img_rel_path) and data_root:
            img_abs_path = os.path.join(data_root, img_rel_path)
        else:
            img_abs_path = img_rel_path

        return img_abs_path

    def _infer_calib_path(self, pts_filepath: str,
                          data_root: str = '') -> str:
        """Infer calib.txt path from LiDAR point cloud path.

        Args:
            pts_filepath (str): Path to LiDAR point cloud file.
            data_root (str): Data root directory.

        Returns:
            str: Absolute path to calib.txt file.
        """
        seq_dir = os.path.dirname(os.path.dirname(pts_filepath))
        calib_path = os.path.join(seq_dir, 'calib.txt')

        if not os.path.isabs(calib_path) and data_root:
            calib_path = os.path.join(data_root, calib_path)

        return calib_path

    def transform(self, input_dict: dict) -> dict:
        """Load calibration and infer image path.

        Args:
            input_dict (dict): Result dict containing 'lidar_points'.

        Returns:
            dict: Updated result dict with:
                - calib (dict): All calibration matrices
                - lidar2img (np.ndarray): LiDAR-to-image projection (3x4)
                - img_path (str): Path to corresponding camera image
        """
        # ------------------------------------------------------------------
        # Step 1: Get paths
        # ------------------------------------------------------------------
        pts_filepath = input_dict['lidar_points']['lidar_path']
        data_root = input_dict.get('data_root', '')

        # ------------------------------------------------------------------
        # Step 2: Infer and set image path (for LoadImageFromFile)
        # ------------------------------------------------------------------
        img_path = self._infer_img_path(pts_filepath, data_root)
        assert os.path.exists(img_path), \
            f'Image file not found: {img_path}. ' \
            f'Please check if {self._camera_to_folder[self.use_camera]}/ ' \
            f'exists in your SemanticKITTI sequence directory.'

        # Set img_path so that LoadImageFromFile can find it
        input_dict['img_path'] = img_path

        # ------------------------------------------------------------------
        # Step 3: Load and parse calib.txt
        # ------------------------------------------------------------------
        calib_path = self._infer_calib_path(pts_filepath, data_root)
        assert os.path.exists(calib_path), \
            f'Calibration file not found: {calib_path}'

        calib_data = self._parse_calib_file(calib_path)

        # ------------------------------------------------------------------
        # Step 4: Reshape matrices
        # ------------------------------------------------------------------
        P0 = calib_data['P0'].reshape(3, 4)
        P1 = calib_data['P1'].reshape(3, 4)
        P2 = calib_data['P2'].reshape(3, 4)
        P3 = calib_data['P3'].reshape(3, 4)
        Tr = calib_data['Tr'].reshape(3, 4)

        # 4x4 homogeneous LiDAR-to-camera transformation
        Tr_homo = np.eye(4, dtype=np.float32)
        Tr_homo[:3, :4] = Tr

        # ------------------------------------------------------------------
        # Step 5: Compute LiDAR-to-image projection matrix
        # ------------------------------------------------------------------
        camera_matrices = {'P0': P0, 'P1': P1, 'P2': P2, 'P3': P3}
        P_cam = camera_matrices[self.use_camera]
        lidar2img = P_cam @ Tr_homo  # (3x4)

        # ------------------------------------------------------------------
        # Step 6: Store results
        # ------------------------------------------------------------------
        input_dict['calib'] = {
            'P0': P0,
            'P1': P1,
            'P2': P2,
            'P3': P3,
            'Tr': Tr,
            'Tr_homo': Tr_homo,
            'lidar2img': lidar2img,
        }
        input_dict['lidar2img'] = lidar2img

        return input_dict

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(use_camera={self.use_camera}, '
        repr_str += f'img_suffix={self.img_suffix})'
        return repr_str