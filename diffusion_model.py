import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# Encode the timestep from int-vector into embedding
class TimestepEmbedding(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.act = nn.SiLU()
    def forward(self, t: torch.Tensor):
        x = self._sinusoidal_time_embedding(t, self.fc1.in_features)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return x
    # Encode the timestep from int-vector to sinusoidal representation
    def _sinusoidal_time_embedding(self, timesteps: torch.Tensor, dim: int):
        half = dim // 2
        freqs = torch.exp(torch.arange(half, device=timesteps.device, dtype=torch.float32) * -(math.log(10000.0) / max(1, half)))
        args = timesteps.float()[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)

        if dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

class CrossAttention(nn.Module):
    def __init__(self, query_dim: int, cond_dim: int, num_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=query_dim, num_heads=num_heads, batch_first=True)
        self.proj_cond = nn.Linear(cond_dim, query_dim)

    def forward(self, query, cond):
        B, C, H, W = query.shape
        query_flat = query.flatten(2).permute(0, 2, 1) # (B, H_Q*W_Q, C)
        cond_flat = cond.flatten(2).permute(0, 2, 1)
        cond_flat = self.proj_cond(cond_flat)  # Match the dimension of query

        out = self.attn(query_flat, cond_flat, cond_flat)[0]
        out = out.permute(0, 2, 1).view(B, C, H, W)
        return out

class ResBlock(nn.Module):
    def __init__(self, c: int, tdim: int):
        """
        c: The number of filter
        tdim: The dimension for time_embedding
        """
        super().__init__()
        num_group = min(8, c)
        self.norm1 = nn.GroupNorm(num_group, c)
        self.conv1 = nn.Conv2d(c, c, 3, padding=1)
        self.norm2 = nn.GroupNorm(num_group, c)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1)
        self.emb = nn.Linear(tdim, 2*c)
        self.act = nn.SiLU()
    def forward(self, x, t_emb):
        gamma, beta = self.emb(t_emb)[:, :, None, None].chunk(2, dim=1)
        h = self.norm1(x)
        h = gamma * h + beta # FiLM
        h = self.conv1(self.act(h))
        h = self.conv2(self.act(self.norm2(h)))
        return x + h

class Down(nn.Module):
    def __init__(self, cin: int, cout: int, tdim: int):
        """
        cin, cout: The channel before/after pooling
        tdim: The dimension for time_embedding
        """
        super().__init__()
        self.block1 = ResBlock(cin, tdim)
        self.proj = nn.Conv2d(cin, cout, kernel_size=2, stride=2)
        self.block2 = ResBlock(cout, tdim)

    def forward(self, x, t):
        x = self.block1(x, t)
        pad_time = x.shape[-2]&1 # For Up to restore
        x = self.proj(x)
        x = self.block2(x, t)
        return x, pad_time

class Up(nn.Module):
    def __init__(self, cin: int, cout: int , tdim: int):
        """
        cin, cout: The channel before/after pooling
        tdim: The dimension for time_embedding
        """
        super().__init__()
        self.block1 = ResBlock(cin, tdim)
        self.proj_nopad = nn.ConvTranspose2d(cin, cout, kernel_size=2, stride=2)
        self.proj_pad = nn.ConvTranspose2d(cin, cout, kernel_size=2, stride=2, output_padding=[1, 0])
        self.block2 = ResBlock(cout, tdim)

    def forward(self, x, t, pad_time):
        x = self.block1(x, t)
        x = self.proj_pad(x) if pad_time else self.proj_nopad(x)
        x = self.block2(x, t)
        return x

class LatentUNet(nn.Module):
    def __init__(self, code_channels: int, base: int, time_dim: int, time_emb_dim: int, model_type):
        """
        code_channels: Code depth
        base: The (base) filter dimension for diffusion model
        time_dim: The dimension to represent timesteps
        time_embed_dim: The dimension to embed the time condition
        model_type: either "DECODER" or "REFINER"
        """
        super().__init__()
        self.model_type = model_type
        self.t_embed = TimestepEmbedding(time_dim, time_emb_dim)
        cin = code_channels * 2  # concatenated noisy z_t and quantized z_q
        self.in_conv = nn.Conv2d(cin, base, 3, padding=1)

        self.cross_attn1 = CrossAttention(base, code_channels, num_heads=1)
        self.cross_attn2 = CrossAttention(base*2, code_channels, num_heads=1)
        self.cross_attn_mid = CrossAttention(base*4, code_channels, num_heads=4)

        # Upsample cond latent to match its original dimension, be careful for output padding if more than 2 pooling layer was passed
        self.cond_upsample = nn.Sequential(
            nn.ConvTranspose2d(code_channels, code_channels, kernel_size=4, stride=2, padding=1),
            #nn.ConvTranspose2d(code_channels, code_channels, kernel_size=4, stride=2, padding=1),
        )

        self.d1 = Down(base, base*2, time_emb_dim)
        self.d2 = Down(base*2, base*4, time_emb_dim)

        self.mid = ResBlock(base*4, time_emb_dim)

        self.u2 = Up(base*4, base*2, time_emb_dim)
        self.u1 = Up(base*2, base, time_emb_dim)

        self.out_conv = nn.Conv2d(base, code_channels, 3, padding=1)
        self.act = nn.SiLU()

    def forward(self, z_t, z_q, t_int):
        """
        z_t, z_q: noisy code and conditioning code (B, C, H, W)
        t_int: (B,) integer timesteps
        """
        t_emb = self.t_embed(t_int)
        if self.model_type == "DECODER":
            z_q = self.cond_upsample(z_q)
        x = torch.cat([z_t, z_q], dim=1)
        x = self.act(self.in_conv(x))

        # Down
        x = x + self.cross_attn1(x, z_q)
        x, pad1 = self.d1(x, t_emb)
        x = x + self.cross_attn2(x, z_q)
        x, pad2 = self.d2(x, t_emb)

        # Mid
        x = x + self.cross_attn_mid(x, z_q)
        x = self.mid(x, t_emb)

        # Up
        x = self.u2(x, t_emb, pad2)
        x = self.u1(x, t_emb, pad1)

        x = self.out_conv(x)
        return x

class LatentDiffusion(nn.Module):
    def __init__(self, code_channels: int, num_filter: int, T: int, time_dim: int, time_emb_dim: int):
        """
        code_channels: Code depth
        num_filter: Filter dimension for diffusion model
        T: The number of time step
        time_dim: The dimension to represent the timestep
        time_emb_dim: The dimension to embed the time condition
        """
        super().__init__()
        self.T = T
        self.model_type = "REFINER" # can be change to "DECODER"
        self.unet = LatentUNet(code_channels=code_channels, base=num_filter, time_dim=time_dim, time_emb_dim=time_emb_dim, model_type=self.model_type)

        """
        Cosine scheduling
        Linear scheduling can be applied with the following
        ```
        betas = torch.linspace(1e-4, 0.02, T, dtype=torch.float32)
        ```
        """
        steps = torch.arange(T + 1, dtype=torch.float32) / T
        alpha_bar = torch.cos((steps + 0.008) / (1 + 0.008) * math.pi / 2) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
        betas = torch.clip(betas, 0.0001, 0.9999)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        # Make the vector to move/store with model
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)

    def p_losses(self, z0, z_q, t): # Training to predict velocity
        noise = torch.randn_like(z0)
        alpha_bar = self.alpha_bars[t].view(-1, 1, 1, 1)

        z_t = alpha_bar.sqrt()*z0 + (1-alpha_bar).sqrt()*noise
        v_pred = self.unet(z_t, z_q, t)
        v_target = alpha_bar.sqrt()*noise - (1-alpha_bar).sqrt()*z0

        return F.mse_loss(v_pred, v_target)

    @torch.no_grad()
    def ddim_sample(self, z_q, steps: int, signal_shape=(100, 128)):
        step_indices = torch.linspace(-1, self.T - 1, steps, device=z_q.device).long().flip(0)
        if self.model_type == "DECODER":
            x =  torch.randn(z_q.shape[0], 1, signal_shape[0], signal_shape[1]).to(z_q.device)
        else:
            x = torch.randn_like(z_q)
        for i in range(len(step_indices)-1):
            t = step_indices[i].item()
            t_prev = step_indices[i + 1].item()
            t_batch = torch.full((z_q.shape[0],), t, device=z_q.device, dtype=torch.long)
            alpha_bar_t = self.alpha_bars[t]
            alpha_bar_prev = self.alpha_bars[t_prev]

            # eta = 0 to fix result
            v_pred = self.unet(x, z_q, t_batch)
            noise_pred = alpha_bar_t.sqrt()*v_pred + (1-alpha_bar_t).sqrt()*x
            x0 = (x - (1-alpha_bar_t).sqrt() * noise_pred) / (alpha_bar_t.sqrt() + 1e-8)
            x = alpha_bar_prev.sqrt()*x0 + (1-alpha_bar_prev).sqrt()*noise_pred 
        return x0
        
class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        super().__init__()
        self.decay = decay
        self.shadow_params = [p.clone().detach() for p in model.parameters()]
        self.temp_params = [p.clone().detach() for p in model.parameters()]
        self.model = model

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
