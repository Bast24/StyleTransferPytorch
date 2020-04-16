import time
import argparse
import torch

####################
# Files parameters #
####################

style_img_path = "./data/style.jpg"
content_img_path = "./data/content.jpg"
output_img = "./data/output.jpg"
model_path = "./data/models/vgg_conv.pth"


#############
# VGG Model #
#############

VGG_CONV_DICT = {
    "conv1_1": {"in_channels": 3, "out_channels": 64, "kernel": 3, "padding": 1,},
    "conv1_2": {"in_channels": 64, "out_channels": 64, "kernel": 3, "padding": 1,},
    "conv2_1": {"in_channels": 64, "out_channels": 128, "kernel": 3, "padding": 1,},
    "conv2_2": {"in_channels": 128, "out_channels": 128, "kernel": 3, "padding": 1,},
    "conv3_1": {"in_channels": 128, "out_channels": 256, "kernel": 3, "padding": 1,},
    "conv3_2": {"in_channels": 256, "out_channels": 256, "kernel": 3, "padding": 1,},
    "conv3_3": {"in_channels": 256, "out_channels": 256, "kernel": 3, "padding": 1,},
    "conv3_4": {"in_channels": 256, "out_channels": 256, "kernel": 3, "padding": 1,},
    "conv4_1": {"in_channels": 256, "out_channels": 512, "kernel": 3, "padding": 1,},
    "conv4_2": {"in_channels": 512, "out_channels": 512, "kernel": 3, "padding": 1,},
    "conv4_3": {"in_channels": 512, "out_channels": 512, "kernel": 3, "padding": 1,},
    "conv4_4": {"in_channels": 512, "out_channels": 512, "kernel": 3, "padding": 1,},
    "conv5_1": {"in_channels": 512, "out_channels": 512, "kernel": 3, "padding": 1,},
    "conv5_2": {"in_channels": 512, "out_channels": 512, "kernel": 3, "padding": 1,},
    "conv5_3": {"in_channels": 512, "out_channels": 512, "kernel": 3, "padding": 1,},
    "conv5_4": {"in_channels": 512, "out_channels": 512, "kernel": 3, "padding": 1,},
}

VGG_POOL_DICT = {
    "pool_1": {"kernel": 2, "stride": 2,},
    "pool_2": {"kernel": 2, "stride": 2,},
    "pool_3": {"kernel": 2, "stride": 2,},
    "pool_4": {"kernel": 2, "stride": 2,},
    "pool_5": {"kernel": 2, "stride": 2,},
}

STYLE_LAYERS = ["relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1"]
STYLE_WEIGHTS = [1e3 / n ** 2 for n in [64, 128, 256, 512, 512]]

CONTENT_LAYERS = ["relu4_2"]
CONTENT_WEIGHTS = [1e0]

###################
# Main parameters #
###################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_iter = 1000
show_every = 50
img_size = 800
