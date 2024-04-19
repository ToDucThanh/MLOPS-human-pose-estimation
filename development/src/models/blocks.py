import torch.nn as nn

from torch.nn import init


def make_OpenPose_block(block_name: str) -> nn.Module:
    blocks = {}

    # Stage 1
    blocks["block1_1"] = [
        {"conv5_1_CPM_L1": [128, 128, 3, 1, 1]},
        {"conv5_2_CPM_L1": [128, 128, 3, 1, 1]},
        {"conv5_3_CPM_L1": [128, 128, 3, 1, 1]},
        {"conv5_4_CPM_L1": [128, 512, 1, 1, 0]},
        {"conv5_5_CPM_L1": [512, 38, 1, 1, 0]},
    ]

    blocks["block1_2"] = [
        {"conv5_1_CPM_L2": [128, 128, 3, 1, 1]},
        {"conv5_2_CPM_L2": [128, 128, 3, 1, 1]},
        {"conv5_3_CPM_L2": [128, 128, 3, 1, 1]},
        {"conv5_4_CPM_L2": [128, 512, 1, 1, 0]},
        {"conv5_5_CPM_L2": [512, 19, 1, 1, 0]},
    ]

    # Stages 2 - 6
    for i in range(2, 7):
        blocks["block%d_1" % i] = [
            {"Mconv1_stage%d_L1" % i: [185, 128, 7, 1, 3]},
            {"Mconv2_stage%d_L1" % i: [128, 128, 7, 1, 3]},
            {"Mconv3_stage%d_L1" % i: [128, 128, 7, 1, 3]},
            {"Mconv4_stage%d_L1" % i: [128, 128, 7, 1, 3]},
            {"Mconv5_stage%d_L1" % i: [128, 128, 7, 1, 3]},
            {"Mconv6_stage%d_L1" % i: [128, 128, 1, 1, 0]},
            {"Mconv7_stage%d_L1" % i: [128, 38, 1, 1, 0]},
        ]

        blocks["block%d_2" % i] = [
            {"Mconv1_stage%d_L2" % i: [185, 128, 7, 1, 3]},
            {"Mconv2_stage%d_L2" % i: [128, 128, 7, 1, 3]},
            {"Mconv3_stage%d_L2" % i: [128, 128, 7, 1, 3]},
            {"Mconv4_stage%d_L2" % i: [128, 128, 7, 1, 3]},
            {"Mconv5_stage%d_L2" % i: [128, 128, 7, 1, 3]},
            {"Mconv6_stage%d_L2" % i: [128, 128, 1, 1, 0]},
            {"Mconv7_stage%d_L2" % i: [128, 19, 1, 1, 0]},
        ]

    cfg_dict = blocks[block_name]

    layers = []

    for i in range(len(cfg_dict)):
        for k, v in cfg_dict[i].items():
            if "pool" in k:
                layers += [nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2])]
            else:
                conv2d = nn.Conv2d(
                    in_channels=v[0],
                    out_channels=v[1],
                    kernel_size=v[2],
                    stride=v[3],
                    padding=v[4],
                )
                layers += [conv2d, nn.ReLU(inplace=True)]

    net = nn.Sequential(*layers[:-1])

    def _initialize_weights_norm(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0.0)

    net.apply(_initialize_weights_norm)
    return net
