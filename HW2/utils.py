import torch

def beta_scheduler(n_timestep=1000, linear_start=1e-4, linear_end=2e-2, device='cuda'):
    betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64).to(device)
    alpha_t = 1 - betas
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
        "beta_t": betas,
        "recipro_sqrt_alpha_t": recipro_sqrt_alpha_t,
        "gamma_t": gamma_t
    }
    
