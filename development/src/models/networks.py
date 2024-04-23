from typing import (
    List,
    Tuple,
)

import torch
import torch.nn as nn
import torchvision

from .blocks import make_OpenPose_block


class OpenPoseNet(nn.Module):
    def __init__(self):
        super(OpenPoseNet, self).__init__()
        self.model0 = OpenPose_Feature()

        self.stage_names = ["block1", "block2", "block3", "block4", "block5", "block6"]

        self.models_1 = nn.ModuleList(
            [make_OpenPose_block(name + "_1") for name in self.stage_names]
        )
        self.models_2 = nn.ModuleList(
            [make_OpenPose_block(name + "_2") for name in self.stage_names]
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], List[torch.Tensor]]:
        out0 = self.model0(x)
        saved_for_loss = []

        out = out0
        for model_1, model_2 in zip(self.models_1, self.models_2):
            out_1 = model_1(out)
            out_2 = model_2(out)
            saved_for_loss.extend([out_1, out_2])
            out = torch.cat([out_1, out_2, out0], 1)

        return (out_1, out_2), saved_for_loss


# class OpenPoseNetOld(nn.Module):
#     def __init__(self):
#         super(OpenPoseNet, self).__init__()
#         self.model0 = OpenPose_Feature()

#         self.model1_1 = make_OpenPose_block("block1_1")
#         self.model2_1 = make_OpenPose_block("block2_1")
#         self.model3_1 = make_OpenPose_block("block3_1")
#         self.model4_1 = make_OpenPose_block("block4_1")
#         self.model5_1 = make_OpenPose_block("block5_1")
#         self.model6_1 = make_OpenPose_block("block6_1")

#         self.model1_2 = make_OpenPose_block("block1_2")
#         self.model2_2 = make_OpenPose_block("block2_2")
#         self.model3_2 = make_OpenPose_block("block3_2")
#         self.model4_2 = make_OpenPose_block("block4_2")
#         self.model5_2 = make_OpenPose_block("block5_2")
#         self.model6_2 = make_OpenPose_block("block6_2")

#     def forward(self, x):
#         # Feature
#         out1 = self.model0(x)

#         # Stage1
#         out1_1 = self.model1_1(out1)  # PAFs
#         out1_2 = self.model1_2(out1)  # confidence heatmap

#         # CStage2
#         out2 = torch.cat([out1_1, out1_2, out1], 1)
#         out2_1 = self.model2_1(out2)
#         out2_2 = self.model2_2(out2)

#         # Stage3
#         out3 = torch.cat([out2_1, out2_2, out1], 1)
#         out3_1 = self.model3_1(out3)
#         out3_2 = self.model3_2(out3)

#         # Stage4
#         out4 = torch.cat([out3_1, out3_2, out1], 1)
#         out4_1 = self.model4_1(out4)
#         out4_2 = self.model4_2(out4)

#         # Stage5
#         out5 = torch.cat([out4_1, out4_2, out1], 1)
#         out5_1 = self.model5_1(out5)
#         out5_2 = self.model5_2(out5)

#         # Stage6
#         out6 = torch.cat([out5_1, out5_2, out1], 1)
#         out6_1 = self.model6_1(out6)
#         out6_2 = self.model6_2(out6)

#         saved_for_loss = []
#         saved_for_loss.append(out1_1)  # PAFs
#         saved_for_loss.append(out1_2)  # confidence heatmap
#         saved_for_loss.append(out2_1)
#         saved_for_loss.append(out2_2)
#         saved_for_loss.append(out3_1)
#         saved_for_loss.append(out3_2)
#         saved_for_loss.append(out4_1)
#         saved_for_loss.append(out4_2)
#         saved_for_loss.append(out5_1)
#         saved_for_loss.append(out5_2)
#         saved_for_loss.append(out6_1)
#         saved_for_loss.append(out6_2)

#         return (out6_1, out6_2), saved_for_loss


class OpenPose_Feature(nn.Module):
    def __init__(self):
        super(OpenPose_Feature, self).__init__()

        vgg19 = torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.DEFAULT)
        model = {}
        model["block0"] = vgg19.features[0:23]

        model["block0"].add_module(
            "23", torch.nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        )
        model["block0"].add_module("24", torch.nn.ReLU(inplace=True))
        model["block0"].add_module(
            "25", torch.nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        )
        model["block0"].add_module("26", torch.nn.ReLU(inplace=True))

        self.model = model["block0"]

    def forward(self, x) -> torch.Tensor:
        outputs = self.model(x)
        return outputs
