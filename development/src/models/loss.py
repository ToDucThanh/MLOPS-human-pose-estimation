from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class OpenPoseLoss(nn.Module):
    def __init__(self):
        super(OpenPoseLoss, self).__init__()

    def forward(
        self,
        saved_for_loss: List[torch.Tensor],
        heatmap_target: torch.Tensor,
        heatmap_mask: torch.Tensor,
        paf_target: torch.Tensor,
        paf_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate the OpenPose loss.

        Args:
            saved_for_loss (List[torch.Tensor]): Output of OpenPoseNet.

            heatmap_target (torch.Tensor):
                Annotation information for heatmaps. Shape: [num_batch, 19, 46, 46].

            heatmap_mask (torch.Tensor):
                Heatmap mask. Shape: [num_batch, 19, 46, 46].

            paf_target (torch.Tensor):
                PAF (Part Affinity Fields) annotation. Shape: [num_batch, 38, 46, 46].

            paf_mask (torch.Tensor):
                PAF mask. Shape: [num_batch, 38, 46, 46].

        Returns:
            torch.Tensor: Total loss.
        """

        total_loss = 0

        for j in range(6):
            # Not count the positions of the mask.
            pred1 = saved_for_loss[2 * j] * paf_mask
            gt1 = paf_target.float() * paf_mask

            # Heatmaps
            pred2 = saved_for_loss[2 * j + 1] * heatmap_mask
            gt2 = heatmap_target.float() * heatmap_mask

            total_loss += F.mse_loss(pred1, gt1, reduction="mean") + F.mse_loss(
                pred2, gt2, reduction="mean"
            )

        return total_loss
