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

from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    RandomHorizontalFlip,
    Resize,
    ToTensor,
)
from datasets import load_dataset
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
from torchinfo import summary

from scheduler import DDIMScheduler
from model import UNet
from utils import save_images, normalize_to_neg_one_to_one, plot_losses, set_seed
import pandas as pd
from torch.utils.data import DataLoader, Dataset, Subset

import torch.distributed as dist
import torch.multiprocessing as mp

import pickle
from datetime import datetime
import pytz

from matplotlib import pyplot as plt
import random

n_timesteps = 1000
noise_scheduler = DDIMScheduler(num_train_timesteps=n_timesteps,
                                    beta_schedule="cosine")


def get_ckpts(checkpoint_dir, last_epoch=False, interval=40, used_one=None):
    checkpoint_ls = sorted(glob.glob(os.path.join(checkpoint_dir, "*.pth")), key=lambda ckpt: int(os.path.basename(ckpt).split("_")[1]))
    checkpoint_ls = [ckpt for ckpt in checkpoint_ls if int(os.path.basename(ckpt).split("_")[1]) % interval == 0]
    if used_one != None:
        ind = (used_one // interval) - 1
        return [checkpoint_ls[ind]]
        
    print('{} checkpoints found'.format(len(checkpoint_ls)))
    if not last_epoch:
        return checkpoint_ls
    else:
        print('using the last epoch only')
        return [checkpoint_ls[-1]]

def d_grad(z, model, gpu, noise=None, l1_loss=True, timesteps=None):
    model.eval()
    
    device = torch.device("cuda:{}".format(gpu) if gpu >= 0 else 'cpu')
    z = z.to(device)    

    loss = 0.0
    
    clean_images = normalize_to_neg_one_to_one(z)

    if noise is None:
        noise = torch.randn(clean_images.shape).to(device)

    if timesteps is None:
        timesteps = torch.randint(0, 
                                noise_scheduler.num_train_timesteps, 
                                (clean_images.shape[0], ),
                                device=device).long()
    noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
    timesteps = timesteps.to(device)
    noise_pred = model(noisy_images, timesteps)["sample"]
    if l1_loss:
        loss += F.l1_loss(noise_pred, noise)
    else:
        loss += F.mse_loss(noise_pred, noise)
        
    params = [ p for p in model.parameters() if p.requires_grad ]
    
    return list(grad(loss, params, create_graph=False))

def save_and_visualize_inf(output_path, influence, helpful, harmful, data_loader, ckpt_iter, n_display=8):
    n_experiments = len(influence)
    os.makedirs(output_path, exist_ok=True)
    outfile = os.path.join(output_path, f'{ckpt_iter}_results')
    dic = {'influence': influence,
            'harmful': harmful,
            'helpful': helpful}
    torch.save(dic, outfile)
    
    X_helpful, X_harmful = [], []
    key = "input" if ('cifar' in output_path or 'tiny' in output_path) else 0
    for j in range(n_experiments):
        for i in range(n_display):
            x_helpful = data_loader.dataset[int(helpful[j][i])][key]
            x_harmful = data_loader.dataset[int(harmful[j][i])][key]
            X_helpful.append(x_helpful)
            X_harmful.append(x_harmful)
    save_image(make_grid(X_helpful), fp=os.path.join(output_path, 'proponents.jpg'))
    save_image(make_grid(X_harmful), fp=os.path.join(output_path, 'opponents.jpg'))
    print('saved {} most/least influential samples for {} images'.format(n_display, len(influence)))

# ========================================================================== #

def diffusion_tracin(dataset, data_loader, z_tests, dict, gpu, retrac=False, last_epoch=False):
    '''
    dict keys: num_avg, interval, batchsize, lr, save_path
    '''
    set_seed(0)
    print("======= (ReTrac) =======" if retrac else "======= (TracIn) =======")

    if 'tiny' in dataset:
        model = UNet(3, image_size=64, hidden_dims=[16, 32, 64, 128])
    elif '64' in dataset:
        model = UNet(3, image_size=64, hidden_dims=[128, 256, 512, 1024])
    else:
        model = UNet(3, image_size=32, hidden_dims=[128, 256, 512, 1024])
        
    device = torch.device("cuda:{}".format(gpu))
    dataset_size = (len(data_loader.dataset) // dict['batchsize']) * dict['batchsize']
    
    pre_ckpt = 0
    influences = [[0.0 for _ in range(dataset_size)] for _ in z_tests]
    ordered_ckpts = get_ckpts(dict['ckpt_dir'], last_epoch=False, interval=dict['interval'])
    
    for checkpoint_name in ordered_ckpts:
        ckpt_iter = int(checkpoint_name.split("/")[-1].split('_')[1])
        ckpt_interval = ckpt_iter - pre_ckpt
        pre_ckpt = ckpt_iter
        print('*' * 20, f' Using Checkpoint: {ckpt_iter} ' + '*' * 20)
        checkpoint = torch.load(os.path.join(checkpoint_name),
                                map_location='cpu')
        model.load_state_dict(checkpoint['model_state'])
        model = model.to(device)
        
        # noise = checkpoint["noise"].to(device)
        timesteps = checkpoint["timesteps"].to(device)
        batch = checkpoint["batch"]
        
        grad_tests = []
        n_avg = dict['num_avg']
        times =  torch.tensor(np.linspace(1, 999, n_avg, dtype=int))
        print("Averaging over {} timesteps...".format(n_avg))
        for num, z_test in enumerate(z_tests):
            z_test = z_test.unsqueeze(0)
            cat_test_grad = []
            for i in range(n_avg):
                grad_test_time_i = []
                for j in range(16):
                    rand_noise = torch.randn(z_test.shape).to(device)
                    grad_test_i = d_grad(z_test, model, gpu, noise=rand_noise, timesteps=times[i])
                    if j == 0:
                        grad_test_time_i = grad_test_i
                    else:
                        grad_test_time_i = [grad_sum_elem + grad for grad_sum_elem, grad in zip(grad_test_time_i, grad_test_i)]
                grad_test_time_i = [grad_sum_elem / 16 for grad_sum_elem in grad_test_time_i]
                if retrac:
                    grad_test_time_i_norm = torch.norm(torch.cat([grad_elem.view(-1) for grad_elem in grad_test_time_i]), 2).item()
                    grad_test_time_i = [grad_test_time_i[b] / grad_test_time_i_norm for b in range(len(grad_test_time_i))]                    

                if i == 0:
                    cat_test_grad = grad_test_time_i
                else:
                    cat_test_grad = [grad_sum_elem + grad for grad_sum_elem, grad in zip(cat_test_grad, grad_test_time_i)]
            
            grad_test = [grad_sum_elem / n_avg for grad_sum_elem in cat_test_grad]
            grad_tests.append(grad_test)
            display_progress("Calc. test grad: ", num, len(z_tests))
            
        for i in range(dataset_size):
            ind = i
            if 'artbench' in dataset:
                z_train = data_loader.dataset[ind][0].unsqueeze(0).to(device)
            else:
                z_train = data_loader.dataset[ind]['input'].unsqueeze(0).to(device)

            in_b = 0
            pos = 0
            for n_batch, b_indices in enumerate(batch):
                for k in range(len(b_indices)):
                    if b_indices[k] == ind:
                        in_b = n_batch
                        pos = k

            batch_ind = in_b
            batch_element_ind = pos
            
            # cur_noise = noise[batch_ind][batch_element_ind].unsqueeze(0)
            cur_noise = torch.randn(z_train.shape).to(device)
            cur_time = timesteps[batch_ind][batch_element_ind]
            
            grad_train = d_grad(z_train, model, gpu, cur_noise, cur_time)
            if retrac:
                grad_train_norm = torch.norm(torch.cat([grad_elem.view(-1) for grad_elem in grad_train]), 2).item()
                grad_train = [(grad_train[i] / grad_train_norm) for i in range(len(grad_train))]
            
            for j in range(len(z_tests)):
                grad_dot_product = sum([torch.sum(k * l).data for k, l in zip(grad_train, grad_tests[j])]).cpu().numpy()
                influences[j][i] += grad_dot_product * dict['lr'] * dict['batchsize'] * ckpt_interval / dataset_size
            
            display_progress("Calc. grad dot product: ", i, dataset_size)
        
        save_path = dict['save_path']
        output_path = f'{save_path}/{ckpt_iter}'
        influences = np.array(influences)
        harmful = np.array([np.argsort(influence) for influence in influences])
        helpful = np.array([x[::-1] for x in harmful])
        
        save_and_visualize_inf(output_path, influences, helpful, harmful, data_loader, ckpt_iter)
            
    influences = np.array(influences)
    harmful = np.array([np.argsort(influence) for influence in influences])
    helpful = np.array([x[::-1] for x in harmful])
    return influences, harmful.tolist(), helpful.tolist()

# ========================================================================== #

def diffusion_tracin_selfinf(dataset, data_loader, dict, gpu, retrac=True, dataset_size=None):
    set_seed(0)
    if dataset_size is None:
        train_dataset_size = len(data_loader.dataset)
    else:
        train_dataset_size = min(dataset_size, len(data_loader.dataset))
    device = torch.device("cuda:{}".format(gpu))

    ckpt_iter = 0
    self_influences = [0.0 for _ in range(train_dataset_size)]
    ordered_checkpoint_list = get_ckpts(dict['ckpt_dir'], last_epoch=False, interval=dict['interval'])

    for checkpoint_name in ordered_checkpoint_list:
        if '64' in dataset:
            model = UNet(3, image_size=64, hidden_dims=[128, 256, 512, 1024])
        else:
            model = UNet(3, image_size=32, hidden_dims=[128, 256, 512, 1024])

        ckpt_iter = int(checkpoint_name.split("/")[-1].split('_')[1])
        checkpoint = torch.load(os.path.join(checkpoint_name),
                                map_location='cpu')
        model.load_state_dict(checkpoint['model_state'])
        model = model.to(device)
        
        ckpt_iter = checkpoint_name.split("/")[-1].split(".")[0]
        print('checkpoint: {}'.format(ckpt_iter))
        
        for i, ind in enumerate(subsample_indices):
            if 'cifar' in dataset:
                z_train = data_loader.dataset[ind]['input'].unsqueeze(0).to(device)
            elif 'artbench' in dataset:
                z_train = data_loader.dataset[ind][0].unsqueeze(0).to(device)
            
            grad_test = []
            n_avg = dict['num_avg']
            times =  torch.tensor(np.linspace(1, 999, n_avg, dtype=int))
            cat_test_grad = []
            for k in range(n_avg):
                grad_test_time_i = []
                for j in range(16):
                    rand_noise = torch.randn(z_train.shape).to(device)
                    grad_test_i = d_grad(z_train, model, gpu, noise=rand_noise, timesteps=times[k])
                    if j == 0:
                        grad_test_time_i = grad_test_i
                    else:
                        grad_test_time_i = [grad_sum_elem + grad for grad_sum_elem, grad in zip(grad_test_time_i, grad_test_i)]
                        
                grad_test_time_i = [grad_sum_elem / 16 for grad_sum_elem in grad_test_time_i]
                if retrac:
                    grad_test_time_i_norm = torch.norm(torch.cat([grad_elem.view(-1) for grad_elem in grad_test_time_i]), 2).item()
                    grad_test_i = [grad_test_time_i[i] / grad_test_time_i_norm for i in range(len(grad_test_time_i))]
                
                if k == 0:
                    cat_test_grad = grad_test_time_i
                else:
                    cat_test_grad = [grad_sum_elem + grad for grad_sum_elem, grad in zip(cat_test_grad, grad_test_time_i)]
            
            grad_test = [grad_sum_elem / n_avg for grad_sum_elem in cat_test_grad]
                
            in_b = 0
            pos = 0
            for j, b in enumerate(batch):
                for k in range(len(b)):
                    if b[k] == ind:
                        in_b = j
                        pos = k
            
            batch_ind = in_b
            batch_element_ind = pos

            cur_noise = noise[batch_ind][batch_element_ind].unsqueeze(0)
            cur_time  = timesteps[batch_ind][batch_element_ind]
            
            grad_train = d_grad(z_train, model, gpu, cur_noise, cur_time)
            if retrac:
                grad_train_norm = torch.norm(torch.cat([grad_elem.view(-1) for grad_elem in grad_train]), 2).item()
                grad_train = [grad_train[i] / grad_train_norm for i in range(len(grad_train))]
            
            grad_dot_product = sum([torch.sum(k * j).data for k, j in zip(grad_train, grad_test)]).cpu().numpy()
            influences[i] += grad_dot_product * dict['lr']
            
            display_progress("Calc. grad dot product: ", i, sub_num)
        
        save_path = dict['save_path']
        output_path = f'{save_path}/{ckpt_iter}'
        self_influences = np.array(influences)
        self_harmful = np.array([np.argsort(influence) for influence in influences])
        self_helpful = np.array([x[::-1] for x in harmful])
        
        save_and_visualize_inf(output_path, self_influences, self_helpful, self_harmful, data_loader, ckpt_iter, n_display=300)
    
    self_influences = np.array(self_influences)
    self_harmful = np.argsort(self_influences) 
    self_helpful = self_harmful[::-1] 
    return self_influences, self_harmful.tolist(), self_helpful.tolist()

def single_sample_varying_timesteps(dataset, data_loader, datset_size, sample_ind, ckpt_path, interval, gpu, save_path):
    if '64' in dataset:
        model = UNet(3, image_size=64, hidden_dims=[128, 256, 512, 1024])
    else:
        model = UNet(3, image_size=32, hidden_dims=[128, 256, 512, 1024])
    
    if 'artbench' in dataset:
        sample = data_loader.dataset[sample_ind][0].unsqueeze(0)
    elif 'cifar' in dataset:
        sample = data_loader.dataset[sample_ind]['input'].unsqueeze(0)
        
    device = torch.device(f'cuda:{gpu}')
    ckpt_lst = get_ckpts(ckpt_path, interval=interval)
    for checkpoint_name in ckpt_lst:
        checkpoint = torch.load(os.path.join(checkpoint_name), map_location='cpu')
        model.load_state_dict(checkpoint['model_state'])
        model = model.to(device)
        timesteps = checkpoint['timesteps'].to(device)
        batch = checkpoint['batch']
        ckpt_iter = checkpoint_name.split("/")[-1].split(".")[0]

        in_b = 0
        pos = 0
        for n_batch, b_indices in enumerate(batch):
            for k in range(len(b_indices)):
                if b_indices[k] == sample_ind:
                    in_b = n_batch
                    pos = k

        batch_ind = in_b
        batch_element_ind = pos
        
        cur_time = timesteps[batch_ind][batch_element_ind]
        
        timesteps_to_use = torch.tensor(np.linspace(1, 999, 100, dtype=int))
        fixed_noise = torch.randn(sample.shape).to(device)
        grad_dists = []
        for index, t in enumerate(timesteps_to_use):
            grad_dist = d_grad(sample, model, gpu, timesteps=t, noise=fixed_noise)
            grad_dist_norm = torch.norm(torch.cat([grad_elem.view(-1) for grad_elem in grad_dist]), 2).item()
            grad_dists.append(grad_dist_norm)
            
            display_progress('Progress: ', index, 100)
            
        plt.figure()
        plt.scatter(np.linspace(1, 999, 100, dtype=int), grad_dists)
        plt.axvline(x=cur_time.item(), color='red', linestyle='--',  label=f'Training Timesteps {cur_time.item()}')
        plt.xlabel('Timesteps')
        plt.ylabel('Gradient Norms')
        plt.legend()
        plt.grid(True, linewidth=0.3, color='gray', alpha=0.7)
        plt.tight_layout()
        save_to = f'{save_path}/{ckpt_iter}'
        os.makedirs(save_to, exist_ok=True)
        plt.savefig(f'{save_to}/dist.png', dpi=300)
        plt.close()

def all_norm_distribution(dataset, data_loader, dataset_size, num_samples, ckpt_path, interval, gpu, save_path):
    random_list = random.sample(range(0, dataset_size), 2000)
    if 'artbench' in dataset:
        samples = [data_loader.dataset[random_list[i]][0].unsqueeze(0) for i in range(len(random_list))]
    elif 'cifar' in dataset:
        samples = [data_loader.dataset[random_list[i]]['input'].unsqueeze(0) for i in range(len(random_list))]
        
    if '64' in dataset:
        model = UNet(3, image_size=64, hidden_dims=[128, 256, 512, 1024])
    else:
        model = UNet(3, image_size=32, hidden_dims=[128, 256, 512, 1024])
    
    device = torch.device(f'cuda:{gpu}')
    ckpt_lst = get_ckpts(ckpt_path, interval=interval)
    for checkpoint_name in ckpt_lst:
        checkpoint = torch.load(os.path.join(checkpoint_name), map_location='cpu')
        model.load_state_dict(checkpoint['model_state'])
        model = model.to(device)
        noise = checkpoint['noise'].to(device)
        timesteps = checkpoint['timesteps'].to(device)
        batch = checkpoint['batch']
        ckpt_iter = checkpoint_name.split("/")[-1].split(".")[0]

        times = []
        norms = []
        for i, sample in enumerate(samples):
            in_b = 0
            pos = 0
            for n_batch, b_indices in enumerate(batch):
                for k in range(len(b_indices)):
                    if b_indices[k] == random_list[i]:
                        in_b = n_batch
                        pos = k

            batch_ind = in_b
            batch_element_ind = pos
            
            cur_noise = noise[batch_ind][batch_element_ind].unsqueeze(0)
            cur_time = timesteps[batch_ind][batch_element_ind]
            times.append(cur_time.item())
            
            grad_sample = d_grad(sample, model, gpu, timesteps=cur_time, noise=cur_noise)
            grad_sample_norm = torch.norm(torch.cat([grad_elem.view(-1) for grad_elem in grad_sample]), 2).item()
            norms.append(grad_sample_norm)
            
            display_progress('Progress: ', i, len(samples))
            
        plt.figure()
        plt.scatter(times, norms)
        plt.xlabel('Timesteps')
        plt.ylabel('Gradient Norms')
        plt.grid(True, linewidth=0.3, color='gray', alpha=0.7)
        plt.tight_layout()
        save_to = f'{save_path}/{ckpt_iter}'
        os.makedirs(save_to, exist_ok=True)
        plt.savefig(f'{save_to}/dist.png', dpi=300)
        plt.close()
