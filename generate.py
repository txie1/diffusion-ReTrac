import argparse
from datetime import datetime

import torch
import torch.nn.functional as F
import numpy as np
import random

import os
from PIL import Image
from utils import set_seed
from torchvision import utils

from scheduler import DDIMScheduler
from model import UNet

n_timesteps = 1000
n_inference_timesteps = 50

def generate_samples(args):
    model = UNet(3, image_size=args.resolution, 
                    hidden_dims=[128, 256, 512, 1024]) 
    noise_scheduler = DDIMScheduler(num_train_timesteps=n_timesteps,
                                    beta_schedule="cosine")

    pretrained = torch.load(args.pretrained_model_path, map_location='cpu')["model_state"]
    model.load_state_dict(pretrained, strict=False)
    
    # for reproducibility
    if args.seed != None:
        set_seed(args.seed)

    if args.gpu < 0:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:{}".format(args.gpu))
    print("generate using gpu:", device)
    model = model.to(device)

    all_samples = []

    with torch.no_grad():
        generated_images = noise_scheduler.generate(
            model,
            num_inference_steps=n_inference_timesteps,
            generator=None,
            eta=0.5,
            use_clipped_model_output=True,
            batch_size=args.eval_batch_size,
            output_type="numpy",
            device=device)

        images = generated_images["sample"]
        all_samples = torch.from_numpy(images)
        images_processed = (images * 255).round().astype("uint8")

        current_date = datetime.today().strftime('%Y%m%d_%H%M%S')
        out_dir = f"./{args.samples_dir}/{current_date}/"
        os.makedirs(out_dir)
        for idx, image in enumerate(images_processed):
            image = Image.fromarray(image)
            image.save(f"{out_dir}/{idx}.jpeg")

        utils.save_image(generated_images["sample_pt"],
                         f"{out_dir}/grid.jpeg",
                         nrow=args.eval_batch_size // 4)
    return all_samples.permute(0, 3, 1, 2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simple script for image generation.")
    parser.add_argument("--samples_dir", type=str, default="test_samples/")
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--pretrained_model_path",
                        type=str,
                        default=None,
                        help="Path to pretrained model")
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--eval_batch_size", type=int, default=12)

    args = parser.parse_args()

    generate_samples(args)
