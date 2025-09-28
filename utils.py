import os
import random
import math
import enum
import numpy as np
import torch
import torch.nn.functional as F

class Refiner(enum.Enum):
    LSTM=0
    GRU=1
    DIFFUSION=2
    TRANSFORMER=3
    MAMBA=4
    HYBRID=5

def _gaussian_kernel(window_size: int, sigma: float, device, channels: int):
    """Create a [channels,1,w,w] Gaussian kernel for grouped conv2d."""
    coords = torch.arange(window_size, device=device) - (window_size - 1) / 2.0
    g = torch.exp(-(coords**2) / (2 * sigma * sigma))
    g = g / g.sum()
    kernel_2d = (g[:, None] @ g[None, :])  # [w,w]
    kernel_2d = kernel_2d / kernel_2d.sum()
    kernel = kernel_2d.expand(channels, 1, window_size, window_size).contiguous()
    return kernel

@torch.no_grad()
def cal_ssim(x: torch.Tensor, y: torch.Tensor, window_size: int = 11, sigma: float = 1.5, K1: float = 0.01, K2: float = 0.03) -> torch.Tensor:
    """
    x,y: [B,C,H,W] on same device. Returns per-sample SSIM [B].
    Uses Gaussian window + classic SSIM formula (Wang et al.).
    """
    assert x.shape == y.shape and x.ndim == 4, "Expect [B,C,H,W] for SSIM."
    B, C, H, W = x.shape
    device = x.device

    # dynamic data range per sample (more robust for standardized EMG)
    # L shape: [B,1,1,1]
    x_max = x.amax(dim=(1,2,3), keepdim=True)
    x_min = x.amin(dim=(1,2,3), keepdim=True)
    y_max = y.amax(dim=(1,2,3), keepdim=True)
    y_min = y.amin(dim=(1,2,3), keepdim=True)
    L = torch.maximum(x_max - x_min, y_max - y_min).clamp(min=1e-8)

    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    window = _gaussian_kernel(window_size, sigma, device, C)

    # reflect padding to avoid border bias
    pad = window_size // 2
    conv = lambda z: F.conv2d(F.pad(z, (pad, pad, pad, pad), mode='reflect'), window, groups=C)

    mu_x = conv(x)
    mu_y = conv(y)
    sigma_x2 = conv(x * x) - mu_x * mu_x
    sigma_y2 = conv(y * y) - mu_y * mu_y
    sigma_xy = conv(x * y) - mu_x * mu_y

    ssim_map = ((2*mu_x*mu_y + C1) * (2*sigma_xy + C2)) / ((mu_x*mu_x + mu_y*mu_y + C1) * (sigma_x2 + sigma_y2 + C2))
    return ssim_map.mean(dim=(1,2,3)) # average over C,H,W -> [B]

@torch.no_grad()
def cal_sisdr(ref: torch.Tensor, est: torch.Tensor):
    """
    Scale-Invariant SDR per channel.
    ref, est: [B, 1, H, W] tensors (same shape as EMG input/output)
    Returns: scalar average SI-SDR across batch and channels
    """
    B, C, H, W = ref.shape
    # Flatten each channel separately: [B, C, H, W] -> [B, C, H*W]
    ref_flat = ref.view(B, C, -1)
    est_flat = est.view(B, C, -1)

    # Projection per channel
    dot = torch.sum(est_flat * ref_flat, dim=-1, keepdim=True)     # [B,C,1]
    ref_energy = torch.sum(ref_flat ** 2, dim=-1, keepdim=True)    # [B,C,1]
    s_target = dot * ref_flat / (ref_energy + 1e-8)                 # [B,C,T]
    e_noise = est_flat - s_target

    sdr = 10 * torch.log10(
        (torch.sum(s_target ** 2, dim=-1) + 1e-8) /
        (torch.sum(e_noise ** 2, dim=-1) + 1e-8)
    )  # [B,C]

    return sdr

# Calculate the loss for VCAE model
def cal_loss(x, recon, mu, logvar, beta):
    MSE = F.mse_loss(recon, x, reduction="mean")
    KL = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + beta * KL

def get_code(model, x):
    # x: (B, 1, 100, 128)
    model.eval()
    with torch.no_grad():
        if model.model_type == "VCAE":
            _, mu, _ = model.encode(x)
            return mu
        else:
            return model.encode(x)

def decode_from_code(model, z):
    # x: (B, 1, T_z, C_z)
    model.eval()
    with torch.no_grad():
        return model.decode(z)

def fix_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
