import torch
import torch.nn as nn
class LSTMRefiner(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 128, num_layers: int = 2, batch_first: bool = True, bidirectional: bool = True):
        super().__init__()
        self.feature_dim = feature_dim  # num_filter * Ch_latent
        self.lstm = nn.LSTM(feature_dim, hidden_dim, num_layers, bidirectional=bidirectional, batch_first=True)
        dir_factor = 2 if bidirectional else 1
        self.conv = nn.Conv2d(1, 1, kernel_size=2, stride=2)
        self.fc = nn.Linear(hidden_dim * dir_factor, feature_dim)  # Project back to feature_dim
        self.convtrans = nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2)

    def forward(self, z_q: torch.Tensor):
        # z_q: (B, C, T_latent, Ch_latent)
        b, c, t_latent, ch_latent = z_q.shape
        # Reshape to sequence: (B, t_latent, c * ch_latent)
        #z_q = self.conv(z_q)
        z_seq = z_q.permute(0, 2, 1, 3).reshape(b, t_latent, c * ch_latent)
        
        # LSTM forward
        out, _ = self.lstm(z_seq)  # (B, t_latent, hidden_dim * dir_factor)
        out = self.fc(out)  # (B, t_latent, feature_dim = c * ch_latent)
        
        # Reshape back to (B, C, T_latent, Ch_latent)
        z_refined = out.view(b, t_latent, c, ch_latent).permute(0, 2, 1, 3)
        #z_refined = self.convtrans(z_refined)
        return z_refined
