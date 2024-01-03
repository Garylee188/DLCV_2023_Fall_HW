import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as transforms
import random
import os
import csv
import numpy as np
import argparse
from PIL import Image

from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from torch.nn import functional as F


myseed = 0
random.seed(myseed)
np.random.seed(myseed)
torch.manual_seed(myseed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)
    torch.cuda.manual_seed(myseed)
    
    
class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, padding=1)
        self.norm1 = nn.BatchNorm2d(out_channel)
        self.activation1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, padding=1)
        self.norm2 = nn.BatchNorm2d(out_channel)
        self.activation2 = nn.ReLU()
    
    def forward(self, x):
        out = self.activation1(self.norm1(self.conv1(x)))
        out = self.activation2(self.norm2(self.conv2(out)))
        return out


class Downsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        
        self.down = nn.Sequential(
            ConvBlock(in_channel, out_channel),
            nn.MaxPool2d(2),
        )
        
    def forward(self, x):
        return self.down(x)
    

class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, 2, 2),
#             nn.Upsample(scale_factor=2),
            ConvBlock(out_channel, out_channel),
            ConvBlock(out_channel, out_channel),
        )
        
    def forward(self, x):
        return self.up(x)

    
class ConditionEmbed(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.embed = nn.Sequential(
            nn.Linear(in_channel, out_channel),
            nn.ReLU(),
            nn.Linear(out_channel, out_channel),
        )

    def forward(self, x):
        x = x.view(-1, self.in_channel)
        x = self.embed(x)
        x = x.view(-1, self.out_channel, 1, 1)  # Reshape to (B, C, 1, 1)
        return x
    
    
class UNet(nn.Module):
    def __init__(self, in_channel, out_channel, num_labels):
        super().__init__()
        
        self.convblock = ConvBlock(in_channel, out_channel)  # (B, out_channel, 28, 28)
        self.down1 = Downsample(out_channel, out_channel)  # (B, out_channel, 14, 14)
        self.down2 = Downsample(out_channel, out_channel*2)  # (B, out_channel*2, 7, 7)
        self.down3 = nn.Sequential(
            nn.MaxPool2d(7),
            nn.ReLU(),
        )  # (B, out_channel*2, 1, 1)
        
        self.label_embed_1c = ConditionEmbed(num_labels, out_channel)
        self.label_embed_2c = ConditionEmbed(num_labels, out_channel*2)
        self.time_embed_1c = ConditionEmbed(1, out_channel)
        self.time_embed_2c = ConditionEmbed(1, out_channel*2)
        
        self.cat_condition_up = nn.Sequential(
            nn.ConvTranspose2d(out_channel*2, out_channel*2, 7, 7),  # if concat -> *3
            nn.GroupNorm(8, out_channel*2),
            nn.ReLU(),
        )  # (B, out_channel*2, 7, 7)
        self.up1 = Upsample(out_channel*4, out_channel)  # (B, out_channel, 14, 14)
        self.up2 = Upsample(out_channel*2, out_channel)  # (B, out_channel, 28, 28)
        
        self.out = nn.Sequential(
            nn.Conv2d(out_channel*2, out_channel, 3, 1, 1),
            nn.GroupNorm(8, out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, in_channel, 3, 1, 1),
        )  # (B, 3, 28, 28)
        
    def forward(self, x, time, labels_each_batch, labels_mask):
        
        # ==================================================================== #
        labels_each_batch = F.one_hot(labels_each_batch, num_classes=10).type(torch.float)
        
        # random mask -> unconditional
        labels_mask = labels_mask[:, None]
        labels_mask = labels_mask.repeat(1, 10)
        labels_mask = (-1*(1-labels_mask))
        
        labels_each_batch = labels_each_batch * labels_mask
        # ==================================================================== #
        
        label_embed_1c = self.label_embed_1c(labels_each_batch)
        label_embed_2c = self.label_embed_2c(labels_each_batch)
        time_embed_1c = self.time_embed_1c(time)
        time_embed_2c = self.time_embed_2c(time)
        
#         feats = []
        
        x0 = self.convblock(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x = self.down3(x2)

        x = self.cat_condition_up(x)   # torch.cat((x, label_embed_2c, time_embed_2c), 1))
        x = self.up1(torch.cat((label_embed_2c * x + time_embed_2c, x2), 1))
        x = self.up2(torch.cat((label_embed_1c * x + time_embed_1c, x1), 1))
        
        x = self.out(torch.cat((x, x0), 1))
        
        return x
    
    
def linear_scheduler(min_beta=1e-4, max_beta=0.02, timestamps=1000, device='cuda'):
    beta_t = min_beta + (max_beta - min_beta) * torch.arange(0, timestamps+1, dtype=torch.float32) / timestamps
    beta_t = beta_t.to(device)
    alpha_t = 1 - beta_t
    alpha_t_hat = torch.cumprod(alpha_t, dim=-1)
    sqrt_alpha_t = torch.sqrt(alpha_t)
    sqrt_alpha_t_hat = torch.sqrt(alpha_t_hat)
    sqrt_one_minus_alpha_t_hat = torch.sqrt(1 - alpha_t_hat)
    recipro_sqrt_alpha_t = 1 / sqrt_alpha_t
    gamma_t = (1 - alpha_t) / sqrt_one_minus_alpha_t_hat
    
    return {
        "sqrt_alpha_t": sqrt_alpha_t,
        "sqrt_alpha_t_hat": sqrt_alpha_t_hat,
        "sqrt_one_minus_alpha_t_hat": sqrt_one_minus_alpha_t_hat, 
        "alpha_t_hat": alpha_t_hat,
        "alpha_t": alpha_t,
        "beta_t": beta_t,
        "recipro_sqrt_alpha_t": recipro_sqrt_alpha_t,
        "gamma_t": gamma_t
    }
    

def save_img(imgs_list, labels_num, labels_list, save_img_dir):
    for idx, img in enumerate(imgs_list):
        img_name = str(labels_list[idx]) + "_" + "{:03d}.png".format(idx // labels_num + 1)
        torchvision.utils.save_image(img, os.path.join(save_img_dir, img_name))
        
        
def part_sample(model, model_path, save_img_dir, sample_size, label_nums, timestamps, time_scheduler, device):
    guide_w = 2
    model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    with torch.no_grad():
        x_t_all = []
        x_t_left_up = []
        img_size = (3, 28, 28)
        
        x = torch.randn(sample_size, *img_size)
                
        labels = torch.tensor(range(label_nums))
        labels = labels.repeat(sample_size//labels.size(0))  # 10 * 100
        
        part_size = 500
        for t in tqdm(range(timestamps, 0, -1)):
            if t > 1:
                z = torch.randn(sample_size, *img_size)
            else:
                z = 0
            
            t_i = torch.tensor([t/timestamps]).to(device)
            t_i = t_i.repeat(part_size, 1, 1, 1)
            t_i = t_i.repeat(2,1,1,1)
                
            for i in range(0, sample_size, part_size):
                x_part = x[i:i+part_size, :, :, :]
                labels_part = labels[i:i+part_size]
                labels_mask = torch.zeros_like(labels_part).to(device)
                
                if t > 1:
                    z_part = z[i:i+part_size, :, :, :]
                    z_part = z_part.to(device)
                else:
                    z_part = 0
                
                x_part = x_part.to(device)
                labels_part = labels_part.to(device)
                
                # double batch
                x_part = x_part.repeat(2,1,1,1)
                labels_part = labels_part.repeat(2)
                labels_mask = labels_mask.repeat(2)
                labels_mask[part_size:] = 1.
                
                eps = model(x_part, t_i, labels_part, labels_mask)
                con_eps = eps[:part_size]
                uncon_eps = eps[part_size:]
                pred_eps = (1 + guide_w) * con_eps - guide_w * uncon_eps
                x_part = x_part[:part_size]

                x_part = time_scheduler["recipro_sqrt_alpha_t"][t] * (x_part - time_scheduler["gamma_t"][t] * pred_eps) + \
                         torch.sqrt(time_scheduler["beta_t"][t]) * z_part
                
                x[i:i+part_size, :, :, :] = x_part.detach().cpu()
            
            x_t_all.append(x)

        save_img(x_t_all[-1], label_nums, labels.detach().cpu().tolist(), save_img_dir)  # for evaluation
        
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_img_dir', type=str, default='./p1_result')
    args = parser.parse_args()
    
    save_img_dir = args.save_img_dir
    
    if not os.path.exists(save_img_dir):
        os.makedirs(save_img_dir)
    
    timestamps = 1000
    sample_size= 1000
    
    model_path = "./diffusion_model.pth"
    model = UNet(in_channel=3, out_channel=128, num_labels=10)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    time_scheduler = linear_scheduler(timestamps=timestamps, device=device)
    
    part_sample(
        model = model,
        model_path = model_path,
        save_img_dir = save_img_dir,
        sample_size = sample_size,
        label_nums = 10,
        timestamps = timestamps,
        time_scheduler = time_scheduler,
        device = device
    )
    
    
if __name__ == "__main__":
    main()
