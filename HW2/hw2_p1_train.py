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
import math
from torch.nn import functional as F
import time


myseed = 0
random.seed(myseed)
np.random.seed(myseed)
torch.manual_seed(myseed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)
    torch.cuda.manual_seed(myseed)


class MNIST_Dataset(Dataset):
    def __init__(self, data_path, csv_file, transform=None):
        self.data_path = data_path
        self.csv_file = csv_file
        self.transform = transform
        
        df = pd.read_csv(self.csv_file)
        self.image_name = df['image_name']
        self.label = df['label']
        
    # return image(tensor) and label
    def __getitem__(self, index):
        image_path = os.path.join(self.data_path, self.image_name[index])
        image = Image.open(image_path)
        label = self.label[index]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

    # return the length of dataset
    def __len__(self):
        return len(self.image_name)


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


def train(model, train_loader, save_model_dir, batch_size, n_epochs, 
          criterion, optimizer, timestamps, time_scheduler, device):
    
    model.to(device)
    model.train()
    for epoch in range(n_epochs):
        print('[Train] Epoch:' + str(epoch+1))
        
        pbar = tqdm(train_loader)
        for batch in pbar:
            imgs, labels = batch
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            # Algorithm 1
            # ========================================================================== #
            t = torch.randint(1, timestamps+1, (batch_size,)).to(device)  # t ~ Uniform(1, timestamps)
            eps = torch.randn_like(imgs)  # eps = noise ~ N(0, 1)
            
            # noisy image at time t, let model learn what noise(eps) is
            x0_coeff = time_scheduler["sqrt_alpha_t_hat"][t].view(batch_size, 1, 1, 1)
            eps_coeff = time_scheduler["sqrt_one_minus_alpha_t_hat"][t].view(batch_size, 1, 1, 1)
            noise_image_t = x0_coeff * imgs + eps_coeff * eps
            # ========================================================================== #
            
            labels_mask = torch.bernoulli(torch.zeros_like(labels)+0.1)
            pred_eps = model(noise_image_t, t/timestamps, labels, labels_mask)
            loss = criterion(pred_eps, eps)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pbar.set_description(f"loss: {loss.item():.4f}")
        
        if (epoch + 1) % 5 == 0:
            model_name = "model_e" + str(epoch+1) + ".pth"
            torch.save(model.state_dict(), os.path.join(save_model_dir, model_name))


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
        
        # Algorithm 2
        # ========================================================================== #
        
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

                x_part = time_scheduler["recipro_sqrt_alpha_t"][t] * (x_part - time_scheduler["gamma_t"][t] * pred_eps) +                          torch.sqrt(time_scheduler["beta_t"][t]) * z_part
                
                x[i:i+part_size, :, :, :] = x_part.detach().cpu()
            # ========================================================================== #
            
            x_t_all.append(x)

        save_img(x_t_all[-1], label_nums, labels.detach().cpu().tolist(), save_img_dir)  # for evaluation


def sample_for_report(model, model_path, save_img_dir, sample_size, label_nums, timestamps, time_scheduler, device):
    guide_w = 2
    model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    with torch.no_grad():
        x_t_all = []
        x_t_left_up = []
        img_size = (3, 28, 28)
        
        x = torch.randn(sample_size, *img_size).to(device)
                
        labels = torch.tensor(range(label_nums)).to(device)
        labels = labels.repeat_interleave(sample_size//labels.shape[0])
        labels_mask = torch.zeros_like(labels).to(device)
        
        labels = labels.repeat(2)
        labels_mask = labels_mask.repeat(2)
        labels_mask[sample_size:] = 1.
        
        # Algorithm 2
        # ========================================================================== #
        
        for t in tqdm(range(timestamps, 0, -1)):
            if t > 1:
                z = torch.randn(sample_size, *img_size).to(device)
            else:
                z = 0
            
            t_i = torch.tensor([t/timestamps]).to(device)
            t_i = t_i.repeat(sample_size, 1, 1, 1)
            t_i = t_i.repeat(2,1,1,1)
            x = x.repeat(2, 1, 1, 1)
            
            eps = model(x, t_i, labels, labels_mask)
            con_eps = eps[:sample_size]
            uncon_eps = eps[sample_size:]
            pred_eps = (1 + guide_w) * con_eps - guide_w * uncon_eps
            x = x[:sample_size]

            x = time_scheduler["recipro_sqrt_alpha_t"][t] * (x - time_scheduler["gamma_t"][t] * pred_eps) +                      torch.sqrt(time_scheduler["beta_t"][t]) * z
            # ========================================================================== #
            
            x_t_left_up.append(x[0].detach().cpu())
            x_t_all.append(x.detach().cpu())
        
        for idx, img in enumerate(x_t_left_up):
            img_name = "0_" + "{:04d}.png".format(idx + 1)
            torchvision.utils.save_image(img, os.path.join(save_img_dir, img_name))  # for report
        
        grid = torchvision.utils.make_grid(x_t_all[-1], nrow=10)
        img_name = "grid_result.png"
        torchvision.utils.save_image(grid, os.path.join(save_img_dir, img_name))  # for report


def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data_dir', type=str, default=r'C:\Users\ntuipmclab\Desktop\DLCV-HW2\2023_hw2_data\hw2_data\digits\mnistm\data')
#     parser.add_argument('--csv_file', type=str, default=r'C:\Users\ntuipmclab\Desktop\DLCV-HW2\2023_hw2_data\hw2_data\digits\mnistm\train.csv')
#     parser.add_argument('--save_dir', type=str, default=r'D:\DLCV-HW2\p1_result')
#     args = parser.parse_args()
    
#     data_dir = args.data_set
#     csv_file = args.csv_file
#     save_dir = args.save_dir

    data_dir = r'C:\Users\ipmc_msi\Desktop\DLCV-HW2\2023_hw2_data\hw2_data\digits\mnistm\data'
    csv_file = r'C:\Users\ipmc_msi\Desktop\DLCV-HW2\2023_hw2_data\hw2_data\digits\mnistm\train.csv'
    save_model_dir = r'E:\DLCV-HW2\p1_result\model'
    save_img_dir = r'E:\DLCV-HW2\p1_result\output'
    save_for_report = r'E:\DLCV-HW2\p1_result\report'
    
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)
    if not os.path.exists(save_img_dir):
        os.makedirs(save_img_dir)
    if not os.path.exists(save_for_report):
        os.makedirs(save_for_report)
    
    # Hyperparameters
    batch_size = 128
    lr         = 0.0001
    n_epochs   = 20
    timestamps = 1000
    sample_size= 1000
    report_sample = 100

    data_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_dataset = MNIST_Dataset(data_dir, csv_file, data_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    model = UNet(in_channel=3, out_channel=128, num_labels=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    time_scheduler = linear_scheduler(timestamps=timestamps, device=device)
    
    train(
        model = model, 
        train_loader = train_loader,
        save_model_dir = save_model_dir,
        batch_size = batch_size,
        n_epochs = n_epochs,
        criterion = criterion,
        optimizer = optimizer,
        timestamps = timestamps,
        time_scheduler = time_scheduler,
        device = device
    )
    
    model_path = os.path.join(save_model_dir, "model_e20.pth")
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
    
#     sample_for_report(
#         model = model,
#         model_path = model_path,
#         save_img_dir = save_for_report,
#         sample_size = report_sample,
#         label_nums = 10,
#         timestamps = timestamps,
#         time_scheduler = time_scheduler,
#         device = device
#     )


if __name__ == "__main__":
    main()
