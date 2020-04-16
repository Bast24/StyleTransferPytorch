import time
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import config as cfg
from models import VGG16, GramMSELoss
from utils import preprocess, make_targets, postprocess


def main():
    n_style = len(cfg.STYLE_LAYERS)
    n_content = len(cfg.CONTENT_LAYERS)

    # Load and process style and content images
    style_img = Image.open(cfg.style_img_path)
    content_img = Image.open(cfg.content_img_path)
    style_img, content_img = preprocess([style_img, content_img])
    style_img, content_img = style_img.to(cfg.device), content_img.to(cfg.device)

    # Image to be optimized
    opt_img = Variable(
        torch.randn(content_img.size()).type_as(content_img.data), requires_grad=True
    ).to(cfg.device)
    # opt_img = Variable(content_img.data.clone(), requires_grad=True)

    # Load model and features
    model = VGG16().to(cfg.device)
    model.load_state_dict(torch.load(cfg.model_path))
    style_targets, content_targets = make_targets(style_img, content_img, model)

    # Make losses and optimizer
    style_losses = [GramMSELoss().to(cfg.device)] * n_style
    content_losses = [nn.MSELoss().to(cfg.device)] * n_content
    optimizer = optim.LBFGS([opt_img])

    # Regroup all losses / targets / layers into one list
    all_layers = cfg.STYLE_LAYERS + cfg.CONTENT_LAYERS
    all_weights = cfg.STYLE_WEIGHTS + cfg.CONTENT_WEIGHTS
    all_losses = style_losses + content_losses
    all_targets = style_targets + content_targets

    # Main loop & optimization
    n_iter = [0]
    start = time.time()
    while n_iter[0] < cfg.max_iter:

        def closure():
            optimizer.zero_grad()
            output = model(opt_img, cfg.STYLE_LAYERS + cfg.CONTENT_LAYERS)
            losses_values = [
                all_weights[i] * all_losses[i](output[i], all_targets[i])
                for i in range(n_style + n_content)
            ]
            total_loss = sum(losses_values)
            total_loss.backward()
            n_iter[0] += 1

            if n_iter[0] % cfg.show_every == 0:
                print(
                    "[{}/{}]  loss: {:.4f}  elapsed time: {:.2f}s".format(
                        n_iter[0], cfg.max_iter, total_loss.item(), time.time() - start,
                    )
                )
            return total_loss

        optimizer.step(closure)

    # Post process and save the output image
    output_image = opt_img.data[0].cpu().squeeze()
    output_image = postprocess(output_image)

    plt.figure()
    plt.imshow(output_image)
    plt.show()

    output_image.save(cfg.output_img)


if __name__ == "__main__":
    main()
