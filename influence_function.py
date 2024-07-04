import torch
import torch.nn.functional as F
from torch_influence import BaseObjective
from torch_influence import AutogradInfluenceModule
from torch_influence import CGInfluenceModule
from torch_influence import LiSSAInfluenceModule


from scheduler import DDIMScheduler
from model import UNet
from utils import save_images, normalize_to_neg_one_to_one, plot_losses, set_seed

import os
import glob
import sys
import argparse
import numpy as np
import torch
from torch.autograd import grad
import torch.nn.functional as F
from utils import display_progress

from torch.nn.utils import clip_grad_norm_
from torchvision.utils import save_image, make_grid
from torchvision import datasets, transforms

from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    RandomHorizontalFlip,
    Resize,
    ToTensor,
)
from datasets import load_dataset, concatenate_datasets
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
from torchinfo import summary

import pandas as pd
from torch.utils.data import DataLoader, Dataset, Subset

import torch.distributed as dist
import torch.multiprocessing as mp

import pickle
from datetime import datetime
import pytz

from matplotlib import pyplot as plt
import random

from artbench_32 import ArtBench10, ArtBench10_subclass

n_timesteps = 1000
noise_scheduler = DDIMScheduler(num_train_timesteps=n_timesteps,
                                    beta_schedule="cosine")

resolution = 32

augmentations = Compose([
                    Resize(resolution, interpolation=InterpolationMode.BILINEAR),
                    CenterCrop(resolution),
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

dataset = load_dataset('cifar10',cache_dir='./dataset/cifar10', split='train')
dataset = dataset.filter(lambda data: data['label'] == 0)
mnist = load_dataset("mnist", split="train")
mnist_0 = mnist.filter(lambda data: data['label'] == 0).rename_column("image", "img")
mnist_0_200 = mnist_0.select(range(200))
mnist_0_200 = mnist_0_200.remove_columns("label")
dataset = concatenate_datasets([dataset, mnist_0_200])
dataset.set_transform(transform)

def loc_in_batch(idx, batch):
    in_b = 0
    pos = 0
    for n_batch, b_indices in enumerate(batch):
        for k in range(len(b_indices)):
            if b_indices[k] == idx:
                in_b = n_batch
                pos = k

    batch_ind = in_b
    batch_element_ind = pos
    
    return batch_ind, batch_element_ind

class CustomDatasetWrapper(Dataset):
    def __init__(self, original_dataset, timesteps, noise_vectors):
        """
        Args:
            original_dataset (Dataset): The original dataset of images.
            timesteps (Tensor): Tensor of timesteps.
            noise_vectors (Tensor): Tensor of noise vectors.
        """
        self.original_dataset = original_dataset
        self.timesteps = timesteps
        self.noise_vectors = noise_vectors
        assert len(original_dataset) == timesteps.size(0) == noise_vectors.size(0), "All inputs must have the same number of elements."

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        idx = int(idx)
        image = self.original_dataset[idx]['input']  # Assuming the original dataset returns an image directly
        timestep = self.timesteps[idx]
        noise_vector = self.noise_vectors[idx]
        return image, timestep, noise_vector
    
device = "cuda:0"

model = UNet(3, image_size=resolution, hidden_dims=[128, 256, 512, 1024])
# checkpoint = torch.load("trained_models/artbench64/ckpt_400_312.pth", map_location='cpu')
checkpoint = torch.load("trained_models/cifar_mnist/ckpt_400_162.pth", map_location='cpu')
model.load_state_dict(checkpoint['model_state'])
model = model.to(device)

timesteps = checkpoint['timesteps'].numpy()
noises = checkpoint['noise'].numpy()
batch = checkpoint['batch']

indices = [loc_in_batch(i, batch) for i in range(len(dataset))]
times = torch.tensor(np.array([timesteps[ind[0], ind[1]] for ind in indices]))
noise = torch.tensor(np.array([noises[ind[0], ind[1]] for ind in indices]))

# Wrap the existing dataset
wrapped_dataset = CustomDatasetWrapper(dataset, times, noise)

# Create DataLoader
dataloader = DataLoader(wrapped_dataset, batch_size=1, shuffle=False)

class MyObjective(BaseObjective):

    def train_outputs(self, model, batch):
        img = batch[0]
        timesteps = batch[1]
        noise = batch[2]
        
        clean_images = normalize_to_neg_one_to_one(img)
        noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
        noise_pred = model(noisy_images, timesteps)["sample"]
        
        return noise_pred

    def train_loss_on_outputs(self, outputs, batch):
        noise = batch[2]
        return F.l1_loss(outputs, noise)
        
    def train_regularization(self, params):
        return 0

    def test_loss(self, model, params, batch):
        img = batch[0]
        clean_images = normalize_to_neg_one_to_one(img)
        
        loss = 0
        times = torch.tensor(np.linspace(1, 999, 20, dtype=int)).to(device)
        
        for i in range(len(times)):
            rand_noise = torch.randn(img.shape).to(device)
            noisy_images = noise_scheduler.add_noise(clean_images, rand_noise, times[i])
            noisy_pred = model(noisy_images, times[i])["sample"]
            loss += F.l1_loss(noisy_pred, rand_noise)
            
        loss /= len(times)
        return loss
    

module = LiSSAInfluenceModule(
    model=model,
    objective=MyObjective(),  
    train_loader=dataloader,
    test_loader=dataloader,
    device=device,
    damp=0.001,
    repeat=30,
    depth=20,
    scale=0.1
)

print("Start Calculation")
res = {}
for i in range(5000, 5008):
    scores = module.influences(list(range(len(dataset))), [i])
    res[i] = scores
    print('i')
torch.save(res, 'zeros.pkl')
