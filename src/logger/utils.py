import io

import matplotlib.pyplot as plt
import PIL
from torchvision.transforms import ToTensor

plt.switch_backend("agg")  # fix RuntimeError: main thread is not in main loop

# Pull secret from kaggle_secrets, .env file and environment.
# Returns the first non-empty source; falls through all of them (the previous
# version stopped at .env even when it returned None, hiding env vars).
def get_secret(name):
    import os

    try:
        from kaggle_secrets import UserSecretsClient

        val = UserSecretsClient().get_secret(name)
        if val:
            return val
    except Exception:
        pass
    try:
        from dotenv import dotenv_values

        val = dotenv_values(".env").get(name)
        if val:
            return val
    except Exception:
        pass
    return os.environ.get(name)

def plot_images(imgs, config):
    """
    Combine several images into one figure.

    Args:
        imgs (Tensor): array of images (B X C x H x W).
        config (DictConfig): hydra experiment config.
    Returns:
        image (Tensor): a single figure with imgs plotted side-to-side.
    """
    # name of each img in the array
    names = config.writer.names
    # figure size
    figsize = config.writer.figsize
    fig, axes = plt.subplots(1, len(names), figsize=figsize)
    for i in range(len(names)):
        # channels must be in the last dim
        img = imgs[i].permute(1, 2, 0)
        axes[i].imshow(img)
        axes[i].set_title(names[i])
        axes[i].axis("off")  # we do not need axis
    # To create a tensor from matplotlib,
    # we need a buffer to save the figure
    buf = io.BytesIO()
    fig.tight_layout()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    # convert buffer to Tensor
    image = ToTensor()(PIL.Image.open(buf))

    plt.close()

    return image
