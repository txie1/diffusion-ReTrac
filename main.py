import os
import time
import glob
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
import random

import torch
from torch.nn import functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torch.autograd import grad
from tqdm.auto import tqdm

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.datasets import VisionDataset

from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
from utils import display_progress, normalize_to_neg_one_to_one, set_seed
from diffusion_tracin import diffusion_tracin, diffusion_tracin_selfinf, single_sample_varying_timesteps, all_norm_distribution
from generate import generate_samples

from datasets import load_dataset, concatenate_datasets
from scheduler import DDIMScheduler
from model import UNet
from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    RandomHorizontalFlip,
    Resize,
    ToTensor,
)
 
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig

def main(args):
    gpu = args.gpu
    device = torch.device("cuda:{}".format(gpu))
    
    if ('tiny' in args.dataset) or ('64' in args.dataset):
        resolution = 64
    else:
        resolution = 32

    # model = UNet(3, image_size=resolution, hidden_dims=[128, 256, 512, 1024])
    model = UNet(3, image_size=resolution, hidden_dims=[16, 32, 64, 128])

        
    if 'artbench' in args.dataset:
        transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),  
            transforms.ToTensor(),           
        ])
    elif ('cifar' in args.dataset) or ('tiny' in args.dataset):
        augmentations = Compose([
            Resize(resolution, interpolation=InterpolationMode.BILINEAR),
            CenterCrop(resolution),
            # RandomHorizontalFlip(),
            ToTensor(),
        ])
        def trans(examples):
            images = [
                augmentations(image.convert("RGB"))
                for image in examples["image"]
                # for image in examples["img"]
            ]
            return {"input": images}
        
        transform = trans
    else:
        print("Dataset not supported")
        exit(1)
        
    if args.dataset == 'artbench_32':
        dataset = ArtBench10_subclass(root='./artbench', train=True, download=True, transform=transform)    
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
            cache_dir='./dataset/cifar10',
            split="train",
        )
        dataset.set_transform(transform)
    elif args.dataset == 'zh-plus/tiny-imagenet':
        dataset = load_dataset(
            'zh-plus/tiny-imagenet',
            cache_dir='dataset/tinyimagenet',
            split="train",
        )
        dataset.set_transform(transform)
    elif args.dataset == 'cifar_mnist':
        dataset = load_dataset('cifar10',cache_dir='./dataset/cifar10', split='train')
        dataset = dataset.filter(lambda data: data['label'] == 0)
        mnist = load_dataset("mnist", split="train")
        mnist_0 = mnist.filter(lambda data: data['label'] == 0).rename_column("image", "img")
        mnist_0_200 = mnist_0.select(range(200))
        mnist_0_200 = mnist_0_200.remove_columns("label")
        dataset = concatenate_datasets([dataset, mnist_0_200])
        dataset.set_transform(transform)
    
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    output_path = args.save_path
    os.makedirs(output_path, exist_ok=True)
    dic = {'ckpt_dir':args.ckpt_dir, 'num_avg':args.num_avg, 'interval':args.interval, 'batchsize':args.batchsize, 'lr':args.lr, 'save_path':output_path}
    if not args.selfinf:
        if args.task == 'train':
            if ('cifar' in args.dataset) or ('tiny' in args.dataset):
                set_seed(16)
                random_samples = random.sample(range(0, len(data_loader.dataset)), args.num_train_samples)
                z_tests = [data_loader.dataset[ind]['input'].to(device) for ind in random_samples]
                # z_tests = [data_loader.dataset[i]['input'].to(device) for i in range(args.num_train_samples)]
            elif 'artbench' in args.dataset:
                z_tests = [data_loader.dataset[i][0].to(device) for i in range(args.num_train_samples)]
            save_image(make_grid(z_tests, nrow=1), fp=os.path.join(output_path, 'train_images.png'))
        else:
            last_ckpt_path = os.path.join(args.ckpt_dir, args.last_ckpt)
            num_samples = args.num_gen_samples
            gen_args = argparse.Namespace(samples_dir=output_path,
                                            resolution=resolution,
                                            pretrained_model_path=last_ckpt_path,
                                            generator=None,
                                            eval_batch_size=num_samples,
                                            seed=1)
            z_tests = generate_samples(gen_args)
            save_image(make_grid(z_tests, nrow=1), fp=os.path.join(output_path, 'gen_images.png'))
            
        influences, harmful, helpful = diffusion_tracin(args.dataset, data_loader, z_tests, dic, args.gpu, retrac=args.retrac, last_epoch=False)
    else:
        influences, harmful, helpful = diffusion_tracin_selfinf(args.dataset, data_loader, dic, args.gpu, retrac=args.retrac, dataset_size=args.selfinf_num)
        
    if args.plot_norm:
        dataset_size = (len(data_loader.dataset) // args.batchsize) * args.batchsize
        single_sample_varying_timesteps(args.dataset, data_loader, dataset_size, args.sample_index, args.ckpt_dir, args.interval, gpu, f'{args.dist_path}/single_sample')
        all_norm_distribution(args.dataset, data_loader, dataset_size, args.num_to_plot, args.ckpt_dir, args.interval, gpu, f'{args.dist_path}/all_samples')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Diffusion-Tracin')
    
    # device
    parser.add_argument('--gpu', type=int, help='which dataset')

    # dataset
    parser.add_argument('--dataset', type=str, help='which dataset')
    parser.add_argument('--num_gen_samples', type=int, default=16, help='number of generated images')
    parser.add_argument('--num_train_samples', type=int, default=16, help='number of train images')
    
    # TracIn
    parser.add_argument('--ckpt_dir', type=str, help='model_ckpt')
    parser.add_argument('--last_ckpt', type=str, default=None, help='final model ckpt for gen sample')
    parser.add_argument('--task', choices=['train', 'gen'], type=str, help='task')
    parser.add_argument('--selfinf', action='store_true', help='calculating selfinf or testinf')
    parser.add_argument('--retrac', action='store_true', help='using retrak or tracin')
    parser.add_argument('--interval', type=int, default=20, help='which ckpt to use')
    parser.add_argument('--num_avg', type=int, default=100, help='number of linspaced timesteps') #50 for tmlr rebuttal
    parser.add_argument('--batchsize', type=int, default=32, help='training batchsize')
    parser.add_argument('--lr', type=float, default=0.0001, help='training learning rate')
    parser.add_argument('--selfinf_num', type=int, default=2000, help='number of samples to calculate selfinf')
    
    # Norm distribution
    parser.add_argument('--ckpt_use', type=str, default=None, help='path to ckpt to calculate the norm dist')
    parser.add_argument('--plot_norm', action='store_true', help='whether to plot the norm distribution')
    parser.add_argument('--sample_index', type=int, default=0, help='sample index to plot the distribution')
    parser.add_argument('--num_to_plot', type=int, default=2000, help='number of random samples to plot the distribution')

    # output
    parser.add_argument('--save_path', type=str, default=None, help='path to save influence')
    parser.add_argument('--dist_path', type=str, default=None, help='path to save norm distribution')
    parser.add_argument('--n_display', type=int, default=8, help='display this number of samples with the most positive/nagetive influences')

    args = parser.parse_args()
    main(args)

'''
Example usage:
python3 main.py --gpu=0 --dataset='cifar_mnist' --plot_norm --ckpt_dir='trained_models/cifar_mnist' --retrac --interval=100 --num_avg=100 --dist_path='test/cifar_mnist'
python3 main.py --gpu=0 --dataset='zh-plus/tiny-imagenet' --ckpt_dir='trained_models/tiny_imagenet' --task='train' --retrac --interval=20 --save_path='influence/tiny_imagenet/retrac'

'''
