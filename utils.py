import os
from datetime import datetime
import sys

from PIL import Image
import torch
import numpy as np
import random
from torchvision import utils
import matplotlib.pyplot as plt

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def plot_losses(losses, epoch, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    plt.plot(losses, label='train')
    plt.legend()
    plt.savefig(f"{out_dir}/{epoch}.png")
    plt.clf()

def save_images(generated_images, epoch, args):
    images = generated_images["sample"]
    images_processed = (images * 255).round().astype("uint8")
    for idx, image in enumerate(images_processed):
        image = Image.fromarray(image)
    os.makedirs(args.samples_dir, exist_ok=True)
    utils.save_image(generated_images["sample_pt"],
                        f"{args.samples_dir}/{epoch}_grid.jpeg",
                        nrow=args.eval_batch_size // 4)

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

def numpy_to_pil(images):
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def match_shape(values, broadcast_array, tensor_format="pt"):
    values = values.flatten()

    while len(values.shape) < len(broadcast_array.shape):
        values = values[..., None]
    if tensor_format == "pt":
        values = values.to(broadcast_array.device)

    return values


def clip(tensor, min_value=None, max_value=None):
    if isinstance(tensor, np.ndarray):
        return np.clip(tensor, min_value, max_value)
    elif isinstance(tensor, torch.Tensor):
        return torch.clamp(tensor, min_value, max_value)

    raise ValueError("Tensor format is not valid is not valid - " \
        f"should be numpy array or torch tensor. Got {type(tensor)}.")


# ============================ #

def display_progress(text, current_step, last_step, enabled=True,
                     fix_zero_start=True):
    """Draws a progress indicator on the screen with the text preceeding the
    progress

    Arguments:
        test: str, text displayed to describe the task being executed
        current_step: int, current step of the iteration
        last_step: int, last possible step of the iteration
        enabled: bool, if false this function will not execute. This is
            for running silently without stdout output.
        fix_zero_start: bool, if true adds 1 to each current step so that the
            display starts at 1 instead of 0, which it would for most loops
            otherwise.
    """
    if not enabled:
        return

    # Fix display for most loops which start with 0, otherwise looks weird
    if fix_zero_start:
        current_step = current_step + 1

    term_line_len = 80
    final_chars = [':', ';', ' ', '.', ',']
    if text[-1:] not in final_chars:
        text = text + ' '
    if len(text) < term_line_len:
        bar_len = term_line_len - (len(text)
                                   + len(str(current_step))
                                   + len(str(last_step))
                                   + len("  / "))
    else:
        bar_len = 30
    filled_len = int(round(bar_len * current_step / float(last_step)))
    bar = '=' * filled_len + '.' * (bar_len - filled_len)

    bar = f"{text}[{bar:s}] {current_step:d} / {last_step:d}"
    if current_step < last_step-1:
        # Erase to end of line and print
        sys.stdout.write("\033[K" + bar + "\r")
    else:
        sys.stdout.write(bar + "\n")

    sys.stdout.flush()


def get_number(name):
    name = name.split('.')[0]
    try:
        name = int(name)
    except:
        pass
    return name
