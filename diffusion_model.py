# latent_diffusion.py
import torch
import torch.nn as nn
import torch.nn.functional as F

DIM = 128
BASE = 128
# ---------- Tiny time embedding ----------
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim=DIM):
        super().__init__()
        self.dim = dim

    def forward(self, t):  # t: [B] long
        half = self.dim // 2
        freq = torch.exp(
            torch.linspace(0, torch.log(torch.tensor(10000.0)), half, device=t.device) * (-1)
        )
        args = t.float().unsqueeze(1) * freq.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:  # pad if odd
            emb = F.pad(emb, (0,1))
        return emb  # [B, dim]

# ---------- A tiny ResBlock with time conditioning ----------
class ResBlock(nn.Module):
    def __init__(self, ch, t_dim=DIM):
        super().__init__()
        #self.norm1 = nn.GroupNorm(8, ch)
        self.norm1 = nn.InstanceNorm2d(ch)
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        #self.norm2 = nn.GroupNorm(8, ch)
        self.norm2 = nn.InstanceNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_dim, ch)
        )
        self.act = nn.SiLU()

    def forward(self, x, t_emb):
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.mlp(t_emb).unsqueeze(-1).unsqueeze(-1)  # FiLM-like add
        h = self.conv2(self.act(self.norm2(h)))
        return x + h

# ---------- Simple latent U-Net (no down/up) ----------
class LatentDenoiser(nn.Module):
    def __init__(self, in_ch, base=BASE, t_dim=DIM):
        super().__init__()
        self.time_emb = SinusoidalTimeEmbedding(t_dim)
        self.in_conv = nn.Conv2d(in_ch, base, 3, padding=1)
        self.block1 = ResBlock(base, t_dim)
        self.block2 = ResBlock(base, t_dim)
        self.block3 = ResBlock(base, t_dim)
        self.block4 = ResBlock(base, t_dim)
        #self.out_norm = nn.GroupNorm(8, base)
        self.out_norm = nn.InstanceNorm2d(base)
        self.out_conv = nn.Conv2d(base, in_ch, 3, padding=1)

    def forward(self, x, t):  # x: [B,C,H,W], t: [B] long
        t_emb = self.time_emb(t)
        h = self.in_conv(x)
        h = self.block1(h, t_emb)
        #h = self.block2(h, t_emb)
        #h = self.block3(h, t_emb)
        #h = self.block4(h, t_emb)
        h = self.out_conv(F.silu(self.out_norm(h)))
        return h  # predict noise

# ---------- Diffusion helpers ----------
class LatentDDPM(nn.Module):
    def __init__(self, denoiser, T, beta_start, beta_end, device, schedule='linear'):
        super().__init__()
        self.denoiser = denoiser
        self.T = T
        self.device = device
        self._beta_schedule(schedule)

    def _beta_schedule(self, schedule):
        if schedule == 'linear':
            betas = torch.linspace(1e-4, 2e-2, self.T)
        else:  # cosine (Nichol & Dhariwal)
            s = 0.008
            steps = self.T + 1
            x = torch.linspace(0, self.T, steps)
            alphas_cumprod = torch.cos(((x / self.T) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alphas_cumprod /= alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = betas.clamp(1e-5, 0.999)

        # --- register schedule buffers
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', 1.0 - betas)
        self.register_buffer('alpha_bars', torch.cumprod(1.0 - betas, dim=0))
        self.register_buffer('alpha_bars_prev',
                             F.pad(self.alpha_bars[:-1], (1, 0), value=1.0))
        self.register_buffer('sqrt_alpha_bars', torch.sqrt(self.alpha_bars))
        self.register_buffer('sqrt_one_minus_alpha_bars', torch.sqrt(1.0 - self.alpha_bars))
        # posterior variance for q(x_{t-1}|x_t, x0)
        self.register_buffer('posterior_variance', betas * (1.0 - self.alpha_bars_prev) / (1.0 - self.alpha_bars))

    @staticmethod
    def _extract(a: torch.Tensor, t: torch.Tensor, x_shape):
        """
        a: [T], t: [B] int64, returns a[t] shaped to broadcast with x
        """
        out = a.gather(0, t).float()
        while out.ndim < len(x_shape):
            out = out.unsqueeze(-1)
        return out

    # ---- forward noising q(x_t | x_0)
    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alpha_bar = self._extract(self.sqrt_alpha_bars, t, x_0.shape)
        sqrt_om_alpha_bar = self._extract(self.sqrt_one_minus_alpha_bars, t, x_0.shape)
        return sqrt_alpha_bar * x_0 + sqrt_om_alpha_bar * noise

    # ---- posterior mean/var for p_theta(x_{t-1} | x_t)
    def p_mean_variance(self, x_t: torch.Tensor, t: torch.Tensor):
        eps = self.denoiser(x_t, t)
        sqrt_alpha_bar = self._extract(self.sqrt_alpha_bars, t, x_t.shape)
        sqrt_om_alpha_bar = self._extract(self.sqrt_one_minus_alpha_bars, t, x_t.shape)

        # predict x0
        x0_pred = (x_t - sqrt_om_alpha_bar * eps) / (sqrt_alpha_bar + 1e-8)
        x0_pred = x0_pred.clamp(-10.0, 10.0)

        alpha = self._extract(self.alphas, t, x_t.shape)
        beta = self._extract(self.betas, t, x_t.shape)
        alpha_bar = self._extract(self.alpha_bars, t, x_t.shape)
        alpha_bar_prev = self._extract(self.alpha_bars_prev, t, x_t.shape)

        # posterior mean
        mean = (
            torch.sqrt(alpha_bar_prev) * beta / (1.0 - alpha_bar) * x0_pred
            + torch.sqrt(alpha) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar) * x_t
        )
        var = self._extract(self.posterior_variance, t, x_t.shape)
        return mean, var

    # ---- ONE reverse step: x_t -> x_{t-1}
    def p_sample_step(self, x_t: torch.Tensor, t: torch.Tensor):
        mean, var = self.p_mean_variance(x_t, t)
        noise = torch.randn_like(x_t)
        # do not add noise at t == 0
        nonzero = (t > 0).float().view(-1, *([1] * (x_t.ndim - 1)))
        return mean + nonzero * torch.sqrt(var) * noise

    def training_loss(self, x_0):
        B = x_0.size(0)
        t = torch.randint(0, self.T, (B,), device=x_0.device, dtype=torch.long)
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)
        pred = self.denoiser(x_t, t)
        return F.mse_loss(pred, noise)

    # ---- full reverse loop from pure noise
    @torch.no_grad()
    def sample(self, shape):
        x_t = torch.randn(shape, device=self.device)
        for step in reversed(range(self.T)):
            t = torch.full((shape[0],), step, device=self.device, dtype=torch.long)
            x_t = self.p_sample_step(x_t, t)
        return x_t
