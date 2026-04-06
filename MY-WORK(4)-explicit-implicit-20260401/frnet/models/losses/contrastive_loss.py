"""
Contrastive Alignment Loss for Explicit Constraint Branch.

Implements the cross-modal contrastive learning loss (Loss_VI) that provides
the "constraint" in the explicit constraint branch.

Core idea (from Fig. 3 & 4):
- Two projection heads φ_V and φ_I map F_Voxel and F'_Image into a shared
  modality-agnostic embedding space, producing Z_V and Z_I.
- Loss_VI (InfoNCE-style) encourages:
  - Positive pairs (same point's Z_V and Z_I) to be close
  - Negative pairs (different semantic class) to be far apart
- sim(·) uses L2-normalized cosine similarity
- τ is a temperature hyperparameter

Formula (Eq. 4):
    Loss_VI = -Σ_i log [ exp(sim(Z_V^i, Z_I^{pos}) / τ) /
                          Σ_{neg} exp(sim(Z_V^i, Z_I^{neg}) / τ) ]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer
from mmdet3d.registry import MODELS
from mmengine.model import BaseModule
from torch import Tensor
from typing import Optional


class ContrastiveProjectionHead(BaseModule):
    """Projection head that maps features to shared embedding space.

    This is the φ_V or φ_I in the paper. A lightweight MLP that projects
    modality-specific features into a modality-agnostic shared space.

    Args:
        in_channels (int): Input feature dimension.
        proj_channels (int): Hidden dimension of projection.
        out_channels (int): Output embedding dimension (shared space).
        norm_cfg (dict): Normalization config.
    """

    def __init__(self,
                 in_channels: int,
                 proj_channels: int = 128,
                 out_channels: int = 64,
                 norm_cfg: dict = dict(type='BN1d', eps=1e-3, momentum=0.01)):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(in_channels, proj_channels, bias=False),
            build_norm_layer(norm_cfg, proj_channels)[1],
            nn.ReLU(inplace=True),
            nn.Linear(proj_channels, out_channels)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (N, C) input features

        Returns:
            z: (N, D) L2-normalized embeddings in shared space
        """
        z = self.projector(x)
        z = F.normalize(z, p=2, dim=-1)  # L2 normalize
        return z


@MODELS.register_module()
class ContrastiveAlignmentLoss(nn.Module):
    """Cross-modal contrastive alignment loss (Loss_VI).

    Encourages corresponding voxel-image feature pairs (same point) to be
    close in shared embedding space, while pushing apart features from
    different semantic classes.

    Negative sampling strategy:
    - For each anchor Z_V^i, negatives are Z_I^j where point j has a
      different semantic label than point i.
    - To keep computation tractable, we randomly sample a fixed number of
      negatives per anchor.

    Args:
        temperature (float): Temperature τ for InfoNCE. Default: 0.07.
        num_negatives (int): Max number of negatives per anchor. Default: 256.
        loss_weight (float): Weight of this loss. Default: 0.1.
        ignore_index (int): Label to ignore. Default: 19.
        sample_ratio (float): Ratio of points to sample for loss computation
            (for efficiency). Default: 0.25.
    """

    def __init__(self,
                 temperature: float = 0.07,
                 num_negatives: int = 256,
                 loss_weight: float = 0.1,
                 ignore_index: int = 19,
                 sample_ratio: float = 0.25):
        super().__init__()
        self.temperature = temperature
        self.num_negatives = num_negatives
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self.sample_ratio = sample_ratio

    def forward(self,
                z_voxel: Tensor,
                z_image: Tensor,
                semantic_labels: Tensor) -> Tensor:
        """Compute contrastive alignment loss.

        Args:
            z_voxel: (N, D) L2-normalized voxel embeddings Z_V
            z_image: (N, D) L2-normalized image embeddings Z_I
            semantic_labels: (N,) semantic label for each point

        Returns:
            loss: scalar contrastive loss
        """
        # Filter out ignored points
        valid_mask = (semantic_labels != self.ignore_index)
        if valid_mask.sum() < 2:
            return z_voxel.sum() * 0.0  # no valid points, return zero loss

        z_v = z_voxel[valid_mask]   # (M, D)
        z_i = z_image[valid_mask]   # (M, D)
        labels = semantic_labels[valid_mask]  # (M,)
        M = z_v.shape[0]

        # Sub-sample for efficiency
        if self.sample_ratio < 1.0:
            num_samples = max(int(M * self.sample_ratio), 2)
            indices = torch.randperm(M, device=z_v.device)[:num_samples]
            z_v_sampled = z_v[indices]          # (S, D)
            z_i_sampled = z_i[indices]          # (S, D)
            labels_sampled = labels[indices]    # (S,)
        else:
            z_v_sampled = z_v
            z_i_sampled = z_i
            labels_sampled = labels
            num_samples = M

        # For each anchor z_v_sampled[i]:
        #   positive: z_i_sampled[i] (same point)
        #   negatives: z_i[j] where labels[j] != labels_sampled[i]

        # Compute positive similarities: sim(Z_V^i, Z_I^{pos})
        # Since both are L2-normalized, sim = dot product = cosine similarity
        pos_sim = (z_v_sampled * z_i_sampled).sum(dim=-1)  # (S,)
        pos_logits = pos_sim / self.temperature  # (S,)

        # Compute negative similarities
        # z_v_sampled: (S, D), z_i: (M, D) -> all pairs: (S, M)
        all_sim = torch.mm(z_v_sampled, z_i.t()) / self.temperature  # (S, M)

        # Mask: negatives are where labels differ
        # labels_sampled: (S,), labels: (M,)
        # neg_mask[i, j] = True if labels_sampled[i] != labels[j]
        neg_mask = (labels_sampled.unsqueeze(1) != labels.unsqueeze(0))  # (S, M)

        # Also mask out the anchor's own positive (exact self)
        # If sampled from same pool, the positive is at indices[i] in the full set
        if self.sample_ratio < 1.0:
            self_mask = torch.zeros(num_samples, M, dtype=torch.bool, device=z_v.device)
            self_mask[torch.arange(num_samples), indices] = True
            neg_mask = neg_mask & (~self_mask)

        # For each anchor, select up to num_negatives negatives
        loss = torch.tensor(0.0, device=z_v.device)
        valid_anchors = 0

        for i in range(num_samples):
            neg_indices_i = neg_mask[i].nonzero(as_tuple=True)[0]

            if neg_indices_i.shape[0] == 0:
                continue  # no negatives for this anchor, skip

            # Random sample negatives if too many
            if neg_indices_i.shape[0] > self.num_negatives:
                perm = torch.randperm(
                    neg_indices_i.shape[0], device=z_v.device)[:self.num_negatives]
                neg_indices_i = neg_indices_i[perm]

            neg_logits_i = all_sim[i, neg_indices_i]  # (K,)

            # InfoNCE: -log(exp(pos) / (exp(pos) + Σ exp(neg)))
            logits = torch.cat([pos_logits[i:i+1], neg_logits_i], dim=0)  # (1+K,)
            target = torch.zeros(1, dtype=torch.long, device=z_v.device)  # positive is index 0
            loss_i = F.cross_entropy(logits.unsqueeze(0), target)

            loss = loss + loss_i
            valid_anchors += 1

        if valid_anchors > 0:
            loss = loss / valid_anchors

        return self.loss_weight * loss


@MODELS.register_module()
class ContrastiveAlignmentLossEfficient(nn.Module):
    """Efficient version of contrastive alignment loss using matrix operations.

    Instead of looping per anchor, uses batch matrix operations.
    Negative sampling: all points with different semantic labels in the batch.

    Args:
        temperature (float): Temperature τ. Default: 0.07.
        loss_weight (float): Loss weight. Default: 0.1.
        ignore_index (int): Label to ignore. Default: 19.
        max_points (int): Max points to use (random subsample). Default: 4096.
    """

    def __init__(self,
                 temperature: float = 0.07,
                 loss_weight: float = 0.1,
                 ignore_index: int = 19,
                 max_points: int = 4096):
        super().__init__()
        self.temperature = temperature
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self.max_points = max_points

    def forward(self,
                z_voxel: Tensor,
                z_image: Tensor,
                semantic_labels: Tensor) -> Tensor:
        """
        Args:
            z_voxel: (N, D) L2-normalized voxel embeddings
            z_image: (N, D) L2-normalized image embeddings
            semantic_labels: (N,) per-point semantic labels

        Returns:
            loss: scalar
        """
        # Filter ignored
        valid_mask = (semantic_labels != self.ignore_index)
        if valid_mask.sum() < 2:
            return z_voxel.sum() * 0.0

        z_v = z_voxel[valid_mask]
        z_i = z_image[valid_mask]
        labels = semantic_labels[valid_mask]
        M = z_v.shape[0]

        # Subsample for memory efficiency
        if M > self.max_points:
            indices = torch.randperm(M, device=z_v.device)[:self.max_points]
            z_v = z_v[indices]
            z_i = z_i[indices]
            labels = labels[indices]
            M = self.max_points

        # Similarity matrix: z_v (query) vs z_i (keys)
        # sim[i, j] = cosine(z_v[i], z_i[j]) / τ
        sim_matrix = torch.mm(z_v, z_i.t()) / self.temperature  # (M, M)

        # Positive mask: diagonal (same point)
        # Negative mask: different semantic label
        label_eq = (labels.unsqueeze(1) == labels.unsqueeze(0))  # (M, M)

        # For InfoNCE: for each anchor i, the positive is z_i[i] (diagonal)
        # negatives are all j where labels[j] != labels[i]
        # We need: -log(exp(sim[i,i]) / (exp(sim[i,i]) + Σ_{j: neg} exp(sim[i,j])))

        # Mask for negatives: different class
        neg_mask = ~label_eq  # (M, M)

        # Also include the positive in the denominator
        # Create log-sum-exp over positives + negatives
        pos_logits = sim_matrix.diag()  # (M,)

        # For denominator: positive + all negatives
        # Set same-class (but not self) entries to very negative so they don't contribute
        # Actually, the cleanest approach:
        # denominator = exp(pos) + Σ_{neg} exp(neg)

        # Mask same-class-but-not-self to -inf
        masked_sim = sim_matrix.clone()
        # Keep: diagonal (positive) + different-class (negatives)
        # Remove: same-class-but-not-diagonal
        same_class_not_self = label_eq & (~torch.eye(M, dtype=torch.bool, device=z_v.device))
        masked_sim[same_class_not_self] = float('-inf')

        # Now masked_sim[i, :] has: sim[i,i] (positive) + negatives + -inf for same-class-not-self
        # The target for cross-entropy is the diagonal index
        targets = torch.arange(M, device=z_v.device)  # (M,)

        loss = F.cross_entropy(masked_sim, targets)

        return self.loss_weight * loss
