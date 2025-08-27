import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

def sinusoidal_time_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        torch.arange(half, device=timesteps.device, dtype=torch.float32) * -(math.log(10000.0) / max(1, half))
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
        self.attn = nn.MultiheadAttention(embed_dim=c, num_heads=4)
    def forward(self, x, t_emb):
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.emb(t_emb)[:, :, None, None]
        h = self.conv2(self.act(self.norm2(h)))
        """
        b, c, h, w = x.shape
        x_flat = x.view(b, c, h*w).permute(2, 0, 1)  # (HW, B, C)
        x_flat = self.attn(x_flat, x_flat, x_flat)[0]
        x = x_flat.permute(1, 2, 0).view(b, c, h, w)
        x = nn.Dropout(p=0.1)(x)
        """
        return x + h

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

class CrossAttention(nn.Module):
    def __init__(self, query_dim: int, cond_dim: int, num_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=query_dim, num_heads=num_heads, batch_first=True)
        self.proj_cond = nn.Linear(cond_dim, query_dim)  # Project cond to match query dim if needed

    def forward(self, query, cond):
        b, c_q, h_q, w_q = query.shape
        query_flat = query.flatten(2).permute(0, 2, 1)

        # cond: (B, C_cond, H_cond, W_cond) -> flatten to (H_cond*W_cond, B, C_q)
        b, c_cond, h_cond, w_cond = cond.shape
        cond_flat = cond.flatten(2).permute(0, 2, 1)  # (H_cond*W_cond, B, C_cond)
        cond_flat = self.proj_cond(cond_flat)  # Project to C_q

        # Cross-attn: query from noisy signal, key/value from cond (latent)
        attended = self.attn(query_flat, cond_flat, cond_flat)[0]
        attended = attended.permute(0, 2, 1).view(b, c_q, h_q, w_q)
        return attended

class LatentUNet(nn.Module):
    def __init__(self, code_channels: int, cond_channels: int, base: int, time_dim: int):
        super().__init__()
        t_embed_dim = base * 4
        self.t_embed = TimestepEmbedding(time_dim, t_embed_dim)
        self.in_conv = nn.Conv2d(code_channels + cond_channels, base, 3, padding=1)

        # Cross-attn layers for conditioning (add at multiple levels)
        self.cross_attn1 = CrossAttention(base, cond_channels)
        self.cross_attn2 = CrossAttention(base * 2, cond_channels)
        self.cross_attn_mid = CrossAttention(base * 4, cond_channels)

        # Upsample cond latent to roughly match signal spatial dims (optional helper for attn)
        self.cond_upsample = nn.Sequential(
            nn.ConvTranspose2d(cond_channels, cond_channels, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(cond_channels, cond_channels, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(cond_channels, cond_channels, kernel_size=4, stride=2, padding=1)
        )

        self.d1 = Down(base, base * 2, t_embed_dim)
        self.d2 = Down(base * 2, base * 4, t_embed_dim)
        self.mid = ResBlock(base * 4, t_embed_dim)
        self.u2 = Up(base * 4, base * 2, t_embed_dim)
        self.u1 = Up(base * 2, base, t_embed_dim)
        self.out_conv = nn.Conv2d(base, code_channels, 3, padding=1)
        self.act = nn.SiLU()

    def forward(self, x_t, z_q, t_int):
        """
        z_t, z_q: (B, C, H, W) where C = code_channels
        t_int: (B,) integer timesteps
        returns predicted noise eps (same shape as z_t)
        """
        t_emb = self.t_embed(t_int)

        # Upsample z_q to closer spatial match (helps attn)
        #z_upsampled = self.cond_upsample(z_q)  # Now ~ (B,4,96,128) or similar; pad/crop if needed
        #z_upsampled = z_q
        z_upsampled = F.interpolate(z_q, size=x_t.shape[2:], mode='bilinear')
        x_cat = torch.cat([x_t, z_upsampled], dim=1)
        x = self.act(self.in_conv(x_cat))

        # Down path with cross-attn
        #x = x + self.cross_attn1(x, z_upsampled)  # Add conditioned features
        x, s1 = self.d1(x, t_emb)
        #x = x + self.cross_attn2(x, z_upsampled)  # Downsampled z could be added, but reuse upsampled for simplicity
        x, s2 = self.d2(x, t_emb)

        x = x + self.cross_attn_mid(x, z_upsampled)
        x = self.mid(x, t_emb)

        # Up path: use recorded shapes in reverse
        x = self.u2(x, t_emb, output_size=s2)
        x = self.u1(x, t_emb, output_size=s1)

        x = self.out_conv(x)
        return x

class LatentDiffusion(nn.Module):
    def __init__(self, code_channels: int, cond_channels: int = 1, num_filter: int = 256, T: int = 1000, time_dim: int = 128, s: float = 0.008):
        super().__init__()
        #betas = torch.linspace(1e-4, 0.02, T, dtype=torch.float32)
        #betas = torch.clip((torch.cos(torch.linspace(0, 1, T+1) * math.pi / 2 + s) / (1 + s)) ** 2, min=1e-5, max=0.999)
        steps = torch.arange(T + 1, dtype=torch.float32) / T
        alpha_bar = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
        betas = torch.clip(betas, 0.0001, 0.9999)

        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.T = T
        self.unet = LatentUNet(code_channels=code_channels, cond_channels=cond_channels, base=num_filter, time_dim=time_dim)

    def q_sample(self, z0, t, noise): # Forward
        # t is a (B,) tensor of indices
        a_bar = self.alpha_bars[t].to(z0.device).view(-1, 1, 1, 1)
        return (a_bar.sqrt() * z0) + ((1 - a_bar).sqrt() * noise)

    def p_losses(self, z0, z_q, t): # Training
        noise = torch.randn_like(z0)
        z_t = self.q_sample(z0, t, noise)
        #noise_pred = self.unet(z_t, z_q, t)
        #"""
        v_pred = self.unet(z_t, z_q, t) # v = sqrt(alpha)*noise - sqrt(1-alpha)*z0

        alpha_bar = self.alpha_bars[t].view(-1, 1, 1, 1)
        v_target = (alpha_bar.sqrt() * noise) - ((1-alpha_bar).sqrt() * z0)

        return F.mse_loss(v_pred, v_target)
        #"""
        #return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def ddim_sample(self, z_q, steps: int = 50, eta: float = 0.5, signal_shape=(100, 128)):#0.0):
        """
        DDIM sampling conditioned on z_q. Returns a denoised latent z_hat.
        Works with arbitrary spatial sizes because Up uses stored output_size.
        """
        T = self.T
        step_indices = torch.linspace(0, T - 1, steps, device=z_q.device).long().flip(0)
        #x = torch.randn_like(z_q)
        x = torch.randn(z_q.shape[0], 1, signal_shape[0], signal_shape[1], device=z_q.device)
        for i in range(len(step_indices) - 1):
            t = step_indices[i].item()
            t_prev = step_indices[i + 1].item()
            t_batch = torch.full((z_q.shape[0],), t, device=z_q.device, dtype=torch.long)
            alpha_bar_t = self.alpha_bars[t].to(z_q.device)
            alpha_bar_prev = self.alpha_bars[t_prev].to(z_q.device)

            #"""
            v_pred = self.unet(x, z_q, t_batch)
            noise_pred = (alpha_bar_t.sqrt()*v_pred + (1-alpha_bar_t).sqrt()*x)# / (1 - alpha_bar_t + 1e-8)
            x0 = (x - (1-alpha_bar_t).sqrt() * noise_pred) / (alpha_bar_t.sqrt() + 1e-8)
            # DDIM update (using derived noise_pred)
            sigma = eta * ((1-alpha_bar_prev)/(1-alpha_bar_t+1e-8) * (1 - alpha_bar_t/alpha_bar_prev) + 1e-8).sqrt()
            dir_coeff = (1 - alpha_bar_prev - sigma**2).clamp(min=0.0).sqrt()
            rand_noise = torch.randn_like(x) if sigma > 0 else torch.zeros_like(x)
            x = alpha_bar_prev.sqrt() * x0 + dir_coeff * noise_pred + sigma * rand_noise

            #dir_xt = (1 - alpha_bar_prev - eta**2 * (1 - alpha_bar_prev) / (1 - alpha_bar_t + 1e-8)).sqrt() * noise_pred
            #noise = torch.randn_like(x) if eta > 0 else torch.zeros_like(x)
            #x = alpha_bar_prev.sqrt() * x0 + dir_xt + eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t + 1e-8)) * noise
            """

            eps = self.unet(x, z_q, t_batch)
            x0 = (x - (1 - alpha_bar_t).sqrt() * eps) / alpha_bar_t.sqrt()
            num = (1 - alpha_bar_prev) / (1 - alpha_bar_t)
            # sigma scalar for this step (DDIM)
            sigma = float(eta * torch.sqrt(num * (1 - (alpha_bar_t / alpha_bar_prev))).item())
            noise = torch.randn_like(x) if sigma > 0 else 0.0
            x = alpha_bar_prev.sqrt() * x0 + (1 - alpha_bar_prev).sqrt() * eps
            if sigma > 0:
                x = x + sigma * noise
        # final step to t=0
        t0 = torch.zeros((z_q.shape[0],), device=z_q.device, dtype=torch.long)
        eps0 = self.unet(x, z_q, t0)
        alpha_bar0 = self.alpha_bars[0]#.to(z_q.device)
        x0 = (x - (1 - alpha_bar0).sqrt() * eps0) / alpha_bar0.sqrt()
            """
        # final step to t=0
        t0 = torch.zeros((z_q.shape[0],), device=z_q.device, dtype=torch.long)
        v_pred = self.unet(x, z_q, t0)
        alpha_bar_0 = self.alpha_bars[0]
        #noise_pred = (alpha_bar_0.sqrt() * v_pred + (1-alpha_bar_0).sqrt() * x) / (1 - alpha_bar_0 + 1e-8)
        noise_pred = alpha_bar_0.sqrt() * v_pred + (1-alpha_bar_0).sqrt() * x
        x0 = (x - (1-alpha_bar_0).sqrt() * noise_pred) / (alpha_bar_0.sqrt() + 1e-8)
        #"""
        return x0
        
# -----------------------
# Uniform quantizer with STE
# -----------------------
#"""
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
"""

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        """
        EMA for model parameters.
        - model: The model to apply EMA to (e.g., unet).
        - decay: EMA decay rate (0.999 is common for diffusion; higher = slower update).
        """
        super().__init__()
        self.decay = decay
        self.temp_params = [p.clone().detach() for p in model.parameters()]
        self.shadow_params = [p.clone().detach() for p in model.parameters()]
        self.model = model  # Reference for swapping

    def update(self):
        with torch.no_grad():
            for shadow_param, param in zip(self.shadow_params, self.model.parameters()):
                shadow_param.data = self.decay * shadow_param.data + (1 - self.decay) * param.data

    def copy_to_model(self):
        with torch.no_grad():
            for original_param, shadow_param, param in zip(self.temp_params, self.shadow_params, self.model.parameters()):
                original_param.data.copy_(param.data)
                param.data.copy_(shadow_param.data)

    def restore_model(self):
        with torch.no_grad():
            for original_param, param in zip(self.temp_params, self.model.parameters()):
                param.data.copy_(original_param.data)
