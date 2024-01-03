import torch
from utils import beta_scheduler
from UNet import UNet
import torchvision
import torchvision.transforms as transforms
import random
import os
import numpy as np
import argparse
from PIL import Image
from tqdm.auto import tqdm


myseed = 0
random.seed(myseed)
np.random.seed(myseed)
torch.manual_seed(myseed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)
    torch.cuda.manual_seed(myseed)
    
    
def ddim_sample(model, noise, eta, device):
    
    scheduler = beta_scheduler(device=device)
    total_time = 1000
    total_steps = 50
    time_steps = total_time // total_steps
    
    sample = noise.to(device)
    model.to(device)
    model.eval()
    
    ddim_timestep_seq = np.array(range(0, total_time, time_steps)) + 1
    ddim_timestep_prev_seq = np.append([0], ddim_timestep_seq[:-1])

    with torch.no_grad():
        for t_id in reversed(range(0, total_time, time_steps)):
            
            if t_id == 0:
                return sample
            
            t = t_id + 1
            t_s = torch.tensor([t]).repeat(sample.size(0), 1, 1, 1).to(device)
            t_prev_id = t_id - time_steps
            
            eps = model(sample, t_s)
            
            pred_x0 = (sample - scheduler["sqrt_one_minus_alpha_t_hat"][t_id] * eps) / scheduler["sqrt_alpha_t_hat"][t_id]
            
            sigma_T = eta * torch.sqrt((1 - scheduler["alpha_t_hat"][t_prev_id]) / (1 - scheduler["alpha_t_hat"][t_id])) * \
                      torch.sqrt(1 - (scheduler["alpha_t_hat"][t_id] / scheduler["alpha_t_hat"][t_prev_id]))
            
            pred_direc_point_xt = torch.sqrt(1 - scheduler["alpha_t_hat"][t_prev_id] - sigma_T**2) * eps
            
            x_prev = scheduler["sqrt_alpha_t_hat"][t_prev_id] * pred_x0 + \
                     pred_direc_point_xt + sigma_T * torch.randn_like(sample)
            
            sample = x_prev
            
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--noises_dir', type=str, default='./2023_hw2_data/hw2_data/face/noise')
    parser.add_argument('--save_dir', type=str, default='./p2_result')
    parser.add_argument('--weight_path', type=str, default='./2023_hw2_data/hw2_data/face')
    args = parser.parse_args()
    
    noises_dir = args.noises_dir
    save_dir = args.save_dir
    weight_path = args.weight_path
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = UNet()
    model.load_state_dict(torch.load(weight_path))
    
    noise_pts = os.listdir(noises_dir)
    
    # Sampling
    for noise_pt in tqdm(noise_pts):
        noise = torch.load(os.path.join(noises_dir, noise_pt))
        result = ddim_sample(model, noise, 0, device)
        result = torch.clamp(result, min=-1., max=1.)
        
        image_name = noise_pt.split('.')[0] + '.png'
        torchvision.utils.save_image(result.cpu(), os.path.join(save_dir, image_name), normalize=True)