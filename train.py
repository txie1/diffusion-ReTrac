import os
import argparse
import random
from itertools import islice

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data.sampler import SubsetRandomSampler

from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    RandomHorizontalFlip,
    Resize,
    ToTensor,
)
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.datasets import VisionDataset

from datasets import load_dataset, concatenate_datasets
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
from torchinfo import summary

from scheduler import DDIMScheduler
from model import UNet
from utils import save_images, normalize_to_neg_one_to_one, plot_losses, set_seed
import pandas as pd
import numpy as np

from artbench_32 import ArtBench10, ArtBench10_subclass


def main(args):
    n_timesteps = args.n_timesteps
    n_inference_timesteps = args.n_inference
    set_seed(args.seed)
    
    model = UNet(3,
                image_size=args.resolution,
                hidden_dims=[128, 256, 512, 1024],
                use_linear_attn=False
            )

    if args.optim == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
        )
    elif args.optim == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.learning_rate
        )
    else:
        print('Optimizer not supported')
        exit(1)
    
    noise_scheduler = DDIMScheduler(num_train_timesteps=n_timesteps,
                                    beta_schedule="cosine")

    if args.pretrained_model_path:
        pretrained = torch.load(args.pretrained_model_path)["model_state"]
        model.load_state_dict(pretrained, strict=False)
    
    if 'artbench' in args.dataset:
        transform = transforms.Compose([
            transforms.Resize((args.resolution, args.resolution)),  
            transforms.ToTensor(),           
        ])
    elif 'cifar' in args.dataset:
        augmentations = Compose([
            Resize(args.resolution, interpolation=InterpolationMode.BILINEAR),
            CenterCrop(args.resolution),
            # RandomHorizontalFlip(),
            ToTensor(),
        ])
        def trans(examples):
            images = [
                augmentations(image.convert("RGB"))
                # for image in examples["image"]
                for image in examples["img"]
            ]
            return {"input": images}
        
        transform = trans
    else:
        print("Dataset not supported")
        exit(1)

    if args.dataset == 'artbench_32':
        dataset = ArtBench10(root='./artbench', train=True, download=True, transform=transform)    
    elif args.dataset == 'artbench_32_subclass':
        dataset = ArtBench10_subclass(root='./artbench', train=True, download=True, transform=transform, labels=[4,8])
    elif args.dataset == 'artbench_64':
        data_path = 'artbench-10-imagefolder-split/train'
        dataset = datasets.ImageFolder(root=data_path, transform=transform)
    elif args.dataset == 'artbench_64_subclass':
        data_path = 'artbench-10-imagefolder-split/train/subclass'
        dataset = datasets.ImageFolder(root=data_path, transform=transform)
    elif args.dataset == 'cifar10':
        dataset = load_dataset(
            'cifar10',
            cache_dir='dataset/cifar10',
            split="train",
        )
        dataset.set_transform(transform)
    elif args.dataset == 'cifar_mnist':
        dataset = load_dataset('cifar10', cache_dir='dataset/cifar10', split='train')
        dataset = dataset.filter(lambda data: data['label'] == 0)
        print("dataset from cache is loaded.")
        mnist = load_dataset("mnist", split="train")
        mnist_0 = mnist.filter(lambda data: data['label'] == 0).rename_column("image", "img")
        mnist_0_200 = mnist_0.select(range(200))
        mnist_0_200 = mnist_0_200.remove_columns("label")
        dataset = concatenate_datasets([dataset, mnist_0_200])
        dataset.set_transform(transform)
    
    truncated_length = (len(dataset) // args.train_batch_size) * args.train_batch_size

    indices = list(range(truncated_length))
    
    device = torch.device(f"cuda:{args.gpu}")
    model = model.to(device)
    # summary(model, [(1, 3, args.resolution, args.resolution), (1, )], verbose=1)
    
    loss_fn = F.l1_loss if args.use_l1_loss else F.mse_loss
    scaler = torch.cuda.amp.GradScaler()
    global_step = 0
    losses = []

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        progress_bar = tqdm(total=(truncated_length // args.train_batch_size))

        progress_bar.set_description(f"Epoch {epoch}")
        losses_log = 0

        random.shuffle(indices)
        train_dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=False)

        noise_list = []
        timesteps_list = []
        batch_list = []

        for step, batch in enumerate(islice(train_dataloader, truncated_length)):
            batch_size = args.train_batch_size
            batch_indices = indices[step * batch_size: (step + 1) * batch_size]

            if len(batch_indices) == 0:
                continue
            batch_list.append(batch_indices)

            if 'cifar' in args.dataset:
                clean_images = torch.stack([dataset[i]['input'] for i in batch_indices]).to(device)
            elif 'artbench' in args.dataset:
                clean_images = torch.stack([dataset[i][0] for i in batch_indices]).to(device)
                
            clean_images = normalize_to_neg_one_to_one(clean_images)

            noise = torch.randn(clean_images.shape).to(device)
            timesteps = torch.randint(0,
                                      noise_scheduler.num_train_timesteps,
                                      (batch_size, ),
                                      device=device).long()

            noisy_images = noise_scheduler.add_noise(clean_images, noise,
                                                     timesteps)

            noise_pred = model(noisy_images, timesteps)["sample"]
            loss = loss_fn(noise_pred, noise)
            loss.backward()

            noise_list.append(noise.detach().clone())
            timesteps_list.append(timesteps.detach().clone())

            if args.use_clip_grad:
                clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            optimizer.zero_grad()

            progress_bar.update(1)
            losses_log += loss.detach().item()
            logs = {
                "loss_avg": losses_log / (step + 1),
                "loss": loss.detach().item(),
                "lr": args.learning_rate,
                "step": global_step
            }

            progress_bar.set_postfix(**logs)
            global_step += 1
        progress_bar.close()
        losses.append(losses_log / (step + 1))

        # Generate sample images for visual inspection
        if epoch % args.save_model_epochs == 0:
            with torch.no_grad():
                generated_images = noise_scheduler.generate(
                    model,
                    num_inference_steps=n_inference_timesteps,
                    generator=None,
                    eta=1.0,
                    use_clipped_model_output=True,
                    batch_size=args.eval_batch_size,
                    output_type="numpy",
                    device=device)

                save_images(generated_images, epoch, args)
                # plot_losses(losses, f"{args.loss_logs_dir}/{epoch}/")
                plot_losses(losses, epoch, f"{args.loss_logs_dir}/")
                
                #-------------------------------#
                # Saves checkpoints
                complete_noise = torch.stack(noise_list)
                complete_timesteps = torch.stack(timesteps_list)

                os.makedirs(args.output_dir, exist_ok=True)
                ckpt_path = f"{args.output_dir}/ckpt_{epoch}_{step}.pth"
                torch.save(
                    {
                        'model_state': model.state_dict(),
                        'timesteps': complete_timesteps,
                        'noise': complete_noise,
                        'batch': batch_list,
                    }, ckpt_path)
                
                torch.save({'losses': losses}, f"{args.output_dir}/losses")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="huggan/flowers-102-categories")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_timesteps", type=int, default=1000)
    parser.add_argument("--n_inference", type=int, default=50)
    parser.add_argument("--optim", type=str, default='adam')
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--dataset_config_name", type=str, default=None)
    parser.add_argument("--train_data_path",
                        type=str,
                        default=None,
                        help="A df containing paths to training images.")
    parser.add_argument("--output_dir",
                        type=str,
                        default="trained_models/ddpm-model-64.pth")
    parser.add_argument("--samples_dir", type=str, default="test_samples/")
    parser.add_argument("--loss_logs_dir", type=str, default="training_logs")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--resolution", type=int, default=32) #64
    parser.add_argument("--train_batch_size", type=int, default=16) #16
    parser.add_argument("--eval_batch_size", type=int, default=32) #32
    parser.add_argument("--num_epochs", type=int, default=2) #100
    parser.add_argument("--save_model_epochs", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--use_clip_grad", type=bool, default=False)
    parser.add_argument("--use_l1_loss", type=bool, default=True)
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--pretrained_model_path",
                        type=str,
                        default=None,
                        help="Path to pretrained model")

    args = parser.parse_args()

    if args.dataset_name is None and args.train_data_path is None:
        raise ValueError(
            "You must specify either a dataset name from the hub or a train data directory."
        )

    main(args)
    
# python3 train.py --gpu=0 --dataset='cifar10' --learning_rate=0.0001 --num_epochs=400 --save_model_epoch=10 --train_batch_size=32 --resolution=32 --output_dir='trained_models/artbench_test' --samples_dir='trained_outputs/artbench_test'

# python3 train.py --gpu=2 --dataset='cifar_mnist' --learning_rate=0.0001 --num_epochs=2 --save_model_epoch=1 --train_batch_size=32 --resolution=32 --output_dir='trained_models/cifar_test' --samples_dir='trained_outputs/cifar_test' --loss_logs_dir="training_logs/cifar_test"
# python3 train.py --dataset='artbench_64_subclass' --learning_rate=0.0001 --num_epochs=400 --save_model_epoch=20 --train_batch_size=32 --resolution=64 --output_dir='trained_models/artbench64' --samples_dir='trained_outputs/artbench64' --loss_logs_dir='training_logs/artbench64' --gpu=1
