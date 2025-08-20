# latent_diffusion.py
import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------
# Time embedding utils
# -----------------------
def sinusoidal_time_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    device = timesteps.device
    half = dim // 2
    freqs = torch.exp(
        torch.arange(half, device=device, dtype=torch.float32) * -(math.log(10000.0) / max(1, half))
    )
    args = timesteps.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb

class TimestepEmbedding(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.act = nn.SiLU()
    def forward(self, t: torch.Tensor):
        x = sinusoidal_time_embedding(t, self.fc1.in_features)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return x

# -----------------------
# ResBlock (unchanged)
# -----------------------
class ResBlock(nn.Module):
    def __init__(self, c: int, tdim: int):
        super().__init__()
        g = min(8, c)
        self.norm1 = nn.GroupNorm(g, c)
        self.conv1 = nn.Conv2d(c, c, 3, padding=1)
        self.norm2 = nn.GroupNorm(g, c)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1)
        self.emb = nn.Linear(tdim, c)
        self.act = nn.SiLU()
    def forward(self, x, t_emb):
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.emb(t_emb)[:, :, None, None]
        h = self.conv2(self.act(self.norm2(h)))
        return x + h

# -----------------------
# Robust Down / Up blocks
# - Down returns (out, pre_shape)
# - Up takes a target output_size and uses interpolate(output_size) to restore exact size
# -----------------------
class Down(nn.Module):
    def __init__(self, cin, cout, tdim):
        """
        cin -> cout, with a pooling step if input spatial dims >= 2.
        Returns (x_after_down, pre_shape) where pre_shape = (H, W) before downsampling.
        """
        super().__init__()
        self.block1 = ResBlock(cin, tdim)
        # Use 1x1 conv to change channels after pooling (or if no pooling necessary)
        self.proj = nn.Conv2d(cin, cout, 1)
        self.block2 = ResBlock(cout, tdim)

    def forward(self, x, t):
        # x: (B, C, H, W)
        x = self.block1(x, t)
        pre_shape = (x.shape[-2], x.shape[-1])  # (H, W) before downsample
        H, W = pre_shape
        if H >= 2 or W >= 2:
            # downsample by roughly factor 2 in each dim using interpolation (floor-halving)
            new_H = H // 2 if H >= 2 else H
            new_W = W // 2 if W >= 2 else W
            # use area pooling via interpolate for stable fractional sizes
            x = F.interpolate(x, size=(new_H, new_W), mode='bilinear', align_corners=False)
        # change channels
        x = self.proj(x)
        x = self.block2(x, t)
        return x, pre_shape

class Up(nn.Module):
    def __init__(self, cin, cout, tdim):
        """
        cin -> cout, expands spatial dims to output_size (pre_shape provided by corresponding Down).
        """
        super().__init__()
        self.block1 = ResBlock(cin, tdim)
        # project channels after upsampling
        self.proj = nn.Conv2d(cin, cout, 1)
        self.block2 = ResBlock(cout, tdim)

    def forward(self, x, t, output_size):
        # output_size: (H, W) to restore exactly
        x = self.block1(x, t)
        # interpolate to exact target size (guarantees matching shapes)
        x = F.interpolate(x, size=output_size, mode='bilinear', align_corners=False)
        x = self.proj(x)
        x = self.block2(x, t)
        return x

# -----------------------
# LatentUNet (uses robust Down/Up)
# -----------------------
class LatentUNet(nn.Module):
    def __init__(self, code_channels: int, base: int = 64, t_embed_dim: int = 256):
        super().__init__()
        self.t_embed = TimestepEmbedding(128, t_embed_dim)
        cin = code_channels * 2  # will concat noisy z_t and quantized z_q
        self.in_conv = nn.Conv2d(cin, base, 3, padding=1)

        # depth = 2 (can be tuned). Use Down/Up pairs that store shapes.
        self.d1 = Down(base, base * 2, t_embed_dim)
        self.d2 = Down(base * 2, base * 4, t_embed_dim)

        self.mid = ResBlock(base * 4, t_embed_dim)

        self.u2 = Up(base * 4, base * 2, t_embed_dim)
        self.u1 = Up(base * 2, base, t_embed_dim)

        self.out_conv = nn.Conv2d(base, code_channels, 3, padding=1)
        self.act = nn.SiLU()

    def forward(self, z_t, z_q, t_int):
        """
        z_t, z_q: (B, C, H, W) where C = code_channels
        t_int: (B,) integer timesteps
        returns predicted noise eps (same shape as z_t)
        """
        t_emb = self.t_embed(t_int)
        x = torch.cat([z_t, z_q], dim=1)
        x = self.act(self.in_conv(x))

        # Down path: record shapes
        x, s1 = self.d1(x, t_emb)   # s1 = (H1, W1)
        x, s2 = self.d2(x, t_emb)   # s2 = (H2, W2)

        x = self.mid(x, t_emb)

        # Up path: use recorded shapes in reverse
        x = self.u2(x, t_emb, output_size=s2)
        x = self.u1(x, t_emb, output_size=s1)

        x = self.out_conv(x)
        return x

# -----------------------
# Diffusion core (buffers + DDIM)
# -----------------------
class LatentDiffusion(nn.Module):
    def __init__(self, code_channels: int, T: int = 1000):
        super().__init__()
        # register schedules as buffers so they move with the module to the device
        betas = torch.linspace(1e-4, 0.02, T, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.T = T
        self.unet = LatentUNet(code_channels=code_channels)

    def q_sample(self, z0, t, noise):
        # t is a (B,) tensor of indices
        a_bar = self.alpha_bars[t].to(z0.device).view(-1, 1, 1, 1)
        return (a_bar.sqrt() * z0) + ((1 - a_bar).sqrt() * noise)

    def p_losses(self, z0, z_q, t):
        noise = torch.randn_like(z0)
        z_t = self.q_sample(z0, t, noise)
        noise_pred = self.unet(z_t, z_q, t)
        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def ddim_sample(self, z_q, steps: int = 10, eta: float = 0.0):
        """
        DDIM sampling conditioned on z_q. Returns a denoised latent z_hat.
        Works with arbitrary spatial sizes because Up uses stored output_size.
        """
        T = self.T
        step_indices = torch.linspace(0, T - 1, steps, device=z_q.device).long().flip(0)
        x = torch.randn_like(z_q)
        for i in range(len(step_indices) - 1):
            t = step_indices[i].item()
            t_prev = step_indices[i + 1].item()
            t_batch = torch.full((z_q.shape[0],), t, device=z_q.device, dtype=torch.long)
            a_bar_t = self.alpha_bars[t].to(z_q.device)
            a_bar_prev = self.alpha_bars[t_prev].to(z_q.device)
            eps = self.unet(x, z_q, t_batch)
            x0 = (x - (1 - a_bar_t).sqrt() * eps) / a_bar_t.sqrt()
            num = (1 - a_bar_prev) / (1 - a_bar_t)
            # sigma scalar for this step (DDIM)
            sigma = float(eta * torch.sqrt(num * (1 - (a_bar_t / a_bar_prev))).item())
            noise = torch.randn_like(x) if sigma > 0 else 0.0
            x = a_bar_prev.sqrt() * x0 + (1 - a_bar_prev).sqrt() * eps
            if sigma > 0:
                x = x + sigma * noise
        # final step to t=0
        t0 = torch.zeros((z_q.shape[0],), device=z_q.device, dtype=torch.long)
        eps0 = self.unet(x, z_q, t0)
        a_bar0 = self.alpha_bars[0].to(z_q.device)
        x0 = (x - (1 - a_bar0).sqrt() * eps0) / a_bar0.sqrt()
        return x0

# -----------------------
# Uniform quantizer with STE
# -----------------------
"""
class Quantizer(nn.Module):
    def __init__(self, num_bits=8, learn_range=False):
        super().__init__()
        self.num_bits = num_bits
        self.levels = 2 ** num_bits

        if learn_range:
            # Learnable global scaling (safer for stability)
            self.register_parameter("scale", nn.Parameter(torch.tensor(0.004)))
        else:
            self.scale = None

    def forward(self, x):
        # Use learnable global range if available, else per-batch dynamic range
        if self.scale is not None:
            scale = torch.abs(self.scale) + 1e-8
        else:
            max_val = x.detach().abs().max()
            scale = max_val + 1e-8

        # Normalize to [-1, 1]
        x_norm = x / scale

        # Quantize to discrete levels in [-1, 1]
        x_q = torch.clamp(x_norm, -1, 1)
        step = 2.0 / (self.levels - 1)
        x_q = torch.round(x_q / step) * step

        # Rescale back
        x_hat = x_q * scale

        return x_hat

"""
class UniformQuantizer(nn.Module):
    def __init__(self, step: float = 1.0, clamp: Optional[float] = None):
        super().__init__()
        self.register_buffer('step', torch.tensor(float(step), dtype=torch.float32))
        self.clamp = float(clamp) if clamp is not None else None

    def _q(self, z):
        s = self.step
        if self.clamp is not None:
            z = torch.clamp(z, -self.clamp, self.clamp)
        return torch.round(z / s) * s

    def forward(self, z: torch.Tensor, hard: bool = False):
        if self.training and not hard:
            y = self._q(z)
            return z + (y - z).detach()  # STE
        else:
            return self._q(z)
#"""
