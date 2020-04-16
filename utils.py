import torch
from torchvision import transforms

import config as cfg
from models import GramMatrix


def preprocess(imgs):
    """Preprocess style and content images for VGG16."""
    transform = transforms.Compose(
        [
            transforms.Resize(cfg.img_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]),
            transforms.Normalize(
                mean=[0.40760392, 0.45795686, 0.48501961], std=[1, 1, 1],
            ),
            transforms.Lambda(lambda x: x.mul_(255)),
        ]
    )
    imgs = [transform(img).unsqueeze(0) for img in imgs]
    return imgs


def postprocess(img):
    """Postprocess the optimized image."""
    transform = transforms.Compose(
        [
            transforms.Lambda(lambda x: x.mul_(1.0 / 255)),
            transforms.Normalize(
                mean=[-0.40760392, -0.45795686, -0.48501961], std=[1, 1, 1],
            ),
            transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]),
        ]
    )
    post_img = transform(img)
    post_img[post_img > 1] = 1
    post_img[post_img < 0] = 0
    post_img = transforms.ToPILImage()(post_img)

    return post_img


def make_targets(style_img, content_img, vgg_model):
    """Returns the style and content targets."""

    style_targets = [
        GramMatrix()(x).detach() for x in vgg_model(style_img, cfg.STYLE_LAYERS)
    ]
    content_targets = [x.detach() for x in vgg_model(content_img, cfg.CONTENT_LAYERS)]

    return style_targets, content_targets
