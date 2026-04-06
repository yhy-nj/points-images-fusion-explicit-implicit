"""
Config for FRNet with Explicit + Implicit Constraint Branches.

This config extends the explicit-only config by:
1. Using FRNetExplicitImplicitBackbone (adds implicit branch)
2. Using FRNetExplicitImplicit segmentor (adds implicit loss)
3. Adding implicit constraint hyperparameters

Usage:
    python tools/train.py configs/frnet/frnet-explicit-implicit-semantickitti_seg.py
"""

_base_ = [
    '../_base_/datasets/semantickitti_seg.py',
    '../_base_/schedules/onecycle-50k.py',
    '../_base_/default_runtime.py'
]

custom_imports = dict(
    imports=[
        'frnet.datasets',
        'frnet.datasets.transforms',
        'frnet.models',
    ],
    allow_failed_imports=False)

# ============================================================
# Model config
# ============================================================
model = dict(
    type='FRNetExplicitImplicit',  # Segmentor with explicit + implicit
    data_preprocessor=dict(
        type='FRNetExplicitImplicitDataPreprocessor',
        H=64, W=512,
        fov_up=3.0, fov_down=-25.0,
        ignore_index=19),
    voxel_encoder=dict(
        type='FRNetVoxelEncoder',
        in_channels=4,
        feat_channels=(64, 128),
        with_distance=False,
        voxel_size=(1, 1, 1),
        point_cloud_range=(-50, -50, -4, 50, 50, 2)),

    # ---- Backbone: FRNet + Explicit + Implicit ----
    backbone=dict(
        type='FRNetExplicitImplicitBackbone',
        in_channels=128,
        point_in_channels=128,
        output_shape=(64, 512),
        depth=34,
        stem_channels=128,
        num_stages=4,
        out_channels=(128, 128, 128, 128),
        strides=(1, 2, 2, 2),
        dilations=(1, 1, 1, 1),
        fuse_channels=(256, 128),
        # Explicit constraint params
        enable_explicit=True,
        explicit_image_channels=128,
        explicit_align_channels=128,
        explicit_out_channels=128,
        explicit_num_samples=9,
        # Implicit constraint params (NEW)
        enable_implicit=True,
        implicit_feat_channels=128,      # Must match image encoder feature dim
        implicit_hidden_channels=128,    # MLP hidden dimension
        implicit_num_mlp_layers=2,       # Number of MLP hidden layers
        implicit_alpha_min=0.01,         # Gaussian opacity pruning threshold
    ),

    decode_head=dict(
        type='FRNetHead',
        channels=128,
        num_classes=20,
        dropout_ratio=0,
        loss_ce=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False,
            class_weight=None,
            loss_weight=1.0),
        loss_lovasz=dict(
            type='LovaszLoss', loss_weight=1.5, reduction='none'),
        loss_boundary=dict(type='BoundaryLoss', loss_weight=1.0),
        conv_seg_kernel_size=1,
        ignore_index=19),

    auxiliary_head=[
        dict(
            type='FrustumHead',
            channels=128,
            num_classes=20,
            dropout_ratio=0,
            loss_ce=dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=False,
                class_weight=None,
                loss_weight=1.0),
            loss_lovasz=dict(
                type='LovaszLoss', loss_weight=1.5, reduction='none'),
            loss_boundary=dict(type='BoundaryLoss', loss_weight=1.0),
            conv_seg_kernel_size=1,
            ignore_index=19),
        dict(
            type='FrustumHead',
            channels=128,
            num_classes=20,
            dropout_ratio=0,
            loss_ce=dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=False,
                class_weight=None,
                loss_weight=1.0),
            loss_lovasz=dict(
                type='LovaszLoss', loss_weight=1.5, reduction='none'),
            loss_boundary=dict(type='BoundaryLoss', loss_weight=1.0),
            conv_seg_kernel_size=1,
            ignore_index=19,
            indices=2),
        dict(
            type='FrustumHead',
            channels=128,
            num_classes=20,
            dropout_ratio=0,
            loss_ce=dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=False,
                class_weight=None,
                loss_weight=1.0),
            loss_lovasz=dict(
                type='LovaszLoss', loss_weight=1.5, reduction='none'),
            loss_boundary=dict(type='BoundaryLoss', loss_weight=1.0),
            conv_seg_kernel_size=1,
            ignore_index=19,
            indices=3),
        dict(
            type='FrustumHead',
            channels=128,
            num_classes=20,
            dropout_ratio=0,
            loss_ce=dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=False,
                class_weight=None,
                loss_weight=1.0),
            loss_lovasz=dict(
                type='LovaszLoss', loss_weight=1.5, reduction='none'),
            loss_boundary=dict(type='BoundaryLoss', loss_weight=1.0),
            conv_seg_kernel_size=1,
            ignore_index=19,
            indices=4),
    ],

    # ---- Loss weights ----
    # λ_imp: weight for implicit constraint loss
    implicit_loss_weight=1.0,
)
