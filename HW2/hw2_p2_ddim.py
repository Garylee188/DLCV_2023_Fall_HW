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
from tqdm import tqdm


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
            
            sigma_T = eta * torch.sqrt((1 - scheduler["alpha_t_hat"][t_prev_id]) / (1 - scheduler["alpha_t_hat"][t_id])) * torch.sqrt(1 - (scheduler["alpha_t_hat"][t_id] / scheduler["alpha_t_hat"][t_prev_id]))
            
            pred_direc_point_xt = torch.sqrt(1 - scheduler["alpha_t_hat"][t_prev_id] - sigma_T**2) * eps
            
            x_prev = scheduler["sqrt_alpha_t_hat"][t_prev_id] * pred_x0 + pred_direc_point_xt + sigma_T * torch.randn_like(sample)
            
            sample = x_prev

def evaluation(generated, truth):
    gen_img = np.array(Image.open(generated))
    true_img = np.array(Image.open(truth))
    
    mse = np.mean((gen_img - true_img) ** 2)
    
    return mse


def interpolation(inter_type, z1, z2, model, save_dir, device):
    
    def slerp(z1, z2, alpha):
        theta = torch.acos(torch.sum(z1 * z2) / (torch.norm(z1) * torch.norm(z2)))
        return torch.sin((1-alpha) * theta) * z1 / torch.sin(theta) + torch.sin(alpha * theta) * z2 / torch.sin(theta)
    
    def linear(z1, z2, alpha):
        return (1-alpha) * z1 + alpha * z2
    
    if not os.path.exists(save_dir):
        os.path.makedirs(save_dir)
    
    alpha_list = torch.arange(0.0, 1.1, 0.1).to(device)
    new_z = []
    for i in range(alpha_list.size(0)):
        if inter_type == 'slerp':
            new_z.append(slerp(z1, z2, alpha_list[i]))
            
        if inter_type == 'linear':
            new_z.append(linear(z1, z2, alpha_list[i]))
    
    new_z = torch.cat(new_z, dim=0)
    inter_result_list = []
    for k in tqdm(range(new_z.size(0))):
        inter_result = ddim_sample(model, new_z[k:k+1, :, :, :], 0, device)
        inter_result = torch.clamp(inter_result, min=-1., max=1.)
        inter_result_list.append(inter_result.squeeze(0))
        
#         image_name = "0_" + "{:02d}.png".format(k)
#         torchvision.utils.save_image(inter_result.cpu(), os.path.join(save_dir, image_name), normalize=True)
    grid = torchvision.utils.make_grid(inter_result_list, nrow=11, padding=2)
    img_name = "grid_result.png"
    torchvision.utils.save_image(grid.cpu(), os.path.join(save_dir, img_name), normalize=True)


def eta_test(eta, noise_list, model, device):
    
    noise_list.to(device)
    eta_results = []
    for i in range(noise_list.size(0)):
        eta_result = ddim_sample(model, noise_list[i:i+1, :, :, :], eta, device)
        eta_result = torch.clamp(eta_result, min=-1., max=1.)
        eta_results.append(eta_result.squeeze(0))
    
    return eta_results


if __name__ == "__main__":
    
    noise_path = r'C:\Users\ipmc_msi\Desktop\DLCV-HW2\2023_hw2_data\hw2_data\face\noise'
    weight_path = r'C:\Users\ipmc_msi\Desktop\DLCV-HW2\2023_hw2_data\hw2_data\face\UNet.pt'
    gt_image_dir = r'C:\Users\ipmc_msi\Desktop\DLCV-HW2\2023_hw2_data\hw2_data\face\GT'
    save_dir = r'E:\DLCV-HW2\p2_result'
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = UNet()
    model.load_state_dict(torch.load(weight_path))
    
    mse_list = []
    noise_pts = os.listdir(noise_path)
    
    # eta Testing
#     z0 = torch.load(os.path.join(noise_path, noise_pts[0]))
#     z1 = torch.load(os.path.join(noise_path, noise_pts[1]))
#     z2 = torch.load(os.path.join(noise_path, noise_pts[2]))
#     z3 = torch.load(os.path.join(noise_path, noise_pts[3]))
#     z_list = torch.cat((z0, z1, z2, z3), dim=0)
    
#     different_eta = []
#     for eta in tqdm(np.arange(0, 1.25, 0.25)):
#         different_eta.extend(eta_test(eta, z_list, model, device))
    
#     grid = torchvision.utils.make_grid(different_eta, nrow=4)
#     img_name = "grid_result.png"
#     torchvision.utils.save_image(grid.cpu(), os.path.join(save_dir, img_name), normalize=True)
    
    
    # Interpolation
#     z1 = torch.load(os.path.join(noise_path, noise_pts[0]))
#     z2 = torch.load(os.path.join(noise_path, noise_pts[1]))
#     interpolation(inter_type='linear', z1=z1, z2=z2, model=model, save_dir=save_dir, device=device)
    
    # Sampling
    for noise_pt in tqdm(noise_pts):
        noise = torch.load(os.path.join(noise_path, noise_pt))
        result = ddim_sample(model, noise, 0, device)
        result = torch.clamp(result, min=-1., max=1.)
        
        image_name = noise_pt.split('.')[0] + '.png'
        torchvision.utils.save_image(result.cpu(), os.path.join(save_dir, image_name), normalize=True)
    
    # Evaluation
#     for img in tqdm(os.listdir(gt_image_dir)):
#         gen_img_path = os.path.join(save_dir, img)
#         gt_img_path = os.path.join(gt_image_dir, img)
#         mse_list.append(evaluation(gen_img_path, gt_img_path))
#     print(sum(mse_list) / len(mse_list))



