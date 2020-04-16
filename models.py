import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision

import config as cfg


class VGG16(nn.Module):
    """VGG16 architecture for extracting filters."""

    def __init__(self):
        super().__init__()
        features = torchvision.models.vgg16(pretrained=True).features

        # The weigths are not learnable
        for param in features.parameters():
            param.require_grad = False

        self.layers = {}

        for name, params in cfg.VGG_CONV_DICT.items():
            self.layers[name] = nn.Conv2d(
                in_channels=params["in_channels"],
                out_channels=params["out_channels"],
                kernel_size=params["kernel"],
                padding=params["padding"],
            )

        for name, params in cfg.VGG_POOL_DICT.items():
            self.layers[name] = nn.MaxPool2d(
                kernel_size=params["kernel"], stride=params["stride"]
            )

        self.conv1_1 = self.layers["conv1_1"]
        self.conv1_2 = self.layers["conv1_2"]
        self.conv2_1 = self.layers["conv2_1"]
        self.conv2_2 = self.layers["conv2_2"]
        self.conv3_1 = self.layers["conv3_1"]
        self.conv3_2 = self.layers["conv3_2"]
        self.conv3_3 = self.layers["conv3_3"]
        self.conv3_4 = self.layers["conv3_4"]
        self.conv4_1 = self.layers["conv4_1"]
        self.conv4_2 = self.layers["conv4_2"]
        self.conv4_3 = self.layers["conv4_3"]
        self.conv4_4 = self.layers["conv4_4"]
        self.conv5_1 = self.layers["conv5_1"]
        self.conv5_2 = self.layers["conv5_2"]
        self.conv5_3 = self.layers["conv5_3"]
        self.conv5_4 = self.layers["conv5_4"]

    def forward(self, x, out_keys):
        out = {}
        out["relu1_1"] = F.relu(self.layers["conv1_1"](x))
        out["relu1_2"] = F.relu(self.layers["conv1_2"](out["relu1_1"]))
        out["pool_1"] = self.layers["pool_1"](out["relu1_2"])

        out["relu2_1"] = F.relu(self.layers["conv2_1"](out["pool_1"]))
        out["relu2_2"] = F.relu(self.layers["conv2_2"](out["relu2_1"]))
        out["pool_2"] = self.layers["pool_2"](out["relu2_2"])

        out["relu3_1"] = F.relu(self.layers["conv3_1"](out["pool_2"]))
        out["relu3_2"] = F.relu(self.layers["conv3_2"](out["relu3_1"]))
        out["relu3_3"] = F.relu(self.layers["conv3_3"](out["relu3_2"]))
        out["relu3_4"] = F.relu(self.layers["conv3_4"](out["relu3_3"]))
        out["pool_3"] = self.layers["pool_3"](out["relu3_4"])

        out["relu4_1"] = F.relu(self.layers["conv4_1"](out["pool_3"]))
        out["relu4_2"] = F.relu(self.layers["conv4_2"](out["relu4_1"]))
        out["relu4_3"] = F.relu(self.layers["conv4_3"](out["relu4_2"]))
        out["relu4_4"] = F.relu(self.layers["conv4_4"](out["relu4_3"]))
        out["pool_4"] = self.layers["pool_4"](out["relu4_4"])

        out["relu5_1"] = F.relu(self.layers["conv5_1"](out["pool_4"]))
        out["relu5_2"] = F.relu(self.layers["conv5_2"](out["relu5_1"]))
        out["relu5_3"] = F.relu(self.layers["conv5_3"](out["relu5_2"]))
        out["relu5_4"] = F.relu(self.layers["conv5_4"](out["relu5_3"]))
        out["pool_5"] = self.layers["pool_5"](out["relu5_4"])

        return [out[key] for key in out_keys]


class GramMatrix(nn.Module):
    """Gram matrix for the style loss."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        b, c, h, w = x.size()
        f = x.view(b, c, h * w)
        g = torch.bmm(f, f.transpose(1, 2))
        g.div_(h * w)
        return g


class GramMSELoss(nn.Module):
    """Style loss."""

    def __init__(self):
        super().__init__()

    def forward(self, inp, gram_target):
        gram_input = GramMatrix()(inp)
        output = nn.MSELoss()(gram_input, gram_target)
        return output


if __name__ == "__main__":
    vgg = VGG16()
