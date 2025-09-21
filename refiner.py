import torch
import torch.nn as nn
class LSTMRefiner(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim = 128, num_layers = 2, bidirectional = True):
        """
        feature_dim: The dimension of lstm input, equals to `num_filter * DIM_C` in code
        hidden_dim: The number of expected hidden dimension
        num_layers: The number of parallel lstm blocks
        """
        super().__init__()
        self.feature_dim = feature_dim
        dir_factor = 2 if bidirectional else 1
        self.lstm = nn.LSTM(feature_dim, hidden_dim, num_layers, bidirectional=bidirectional, batch_first=True)
        self.fc = nn.Linear(hidden_dim * dir_factor, feature_dim)  # Project back to feature_dim

    def forward(self, z_q: torch.Tensor):
        b, c, t, ch = z_q.shape
        z_seq = z_q.permute(0, 2, 1, 3).reshape(b, t, c * ch) # (BATCH_SIZE, DIM_T, feature_dim)
        out, _ = self.lstm(z_seq)  # (BATCH_SIZE, DIM_T, hidden_dim * dir_factor)
        out = self.fc(out)  # (BATCH_SIZE, DIM_T, feature_dim)
        out = out.view(b, t, c, ch).permute(0, 2, 1, 3) # (BATCH_SIZE, num_filter, DIM_T, DIM_C)
        return out

class GRURefiner(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim=128, num_layers=2, bidirectional=True):
        super().__init__()
        self.feature_dim = feature_dim
        dir_factor = 2 if bidirectional else 1
        self.gru = nn.GRU(feature_dim, hidden_dim, num_layers, bidirectional=bidirectional, batch_first=True)
        self.fc = nn.Linear(hidden_dim * dir_factor, feature_dim)

    def forward(self, z_q: torch.Tensor):
        b, c, t, ch = z_q.shape
        z_seq = z_q.permute(0, 2, 1, 3).reshape(b, t, c * ch) # (BATCH_SIZE, DIM_T, feature_dim)
        out, _ = self.gru(z_seq)  # (BATCH_SIZE, DIM_T, hidden_dim * dir_factor)
        out = self.fc(out)  # (BATCH_SIZE, DIM_T, feature_dim)
        out = out.view(b, t, c, ch).permute(0, 2, 1, 3) # (BATCH_SIZE, num_filter, DIM_T, DIM_C)
        return out

class TransformerRefiner(nn.Module):
    def __init__(self, feature_dim: int, num_layers=2, num_heads=4, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.pos_encoder = nn.Parameter(torch.randn(1, 1, feature_dim))  # Simple learnable pos emb
        encoder_layer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(feature_dim, feature_dim)  # Optional projection

    def forward(self, z_q: torch.Tensor):
        b, c, t, ch = z_q.shape
        z_seq = z_q.permute(0, 2, 1, 3).reshape(b, t, c * ch)  # (B, T_z, feature_dim)
        z_seq = z_seq + self.pos_encoder.expand(b, t, -1)  # Add pos
        out = self.transformer(z_seq)  # (B, T_z, feature_dim)
        out = self.fc(out)
        return out.view(b, t, c, ch).permute(0, 2, 1, 3)
#"""
from mamba_ssm import Mamba  # Assume you pip install mamba-ssm or copy impl

class MambaRefiner(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim=128, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([Mamba(d_model=feature_dim, d_state=hidden_dim, d_conv=3, expand=2) for _ in range(num_layers)])
        self.fc = nn.Linear(feature_dim, feature_dim)

    def forward(self, z_q: torch.Tensor):
        b, c, t, ch = z_q.shape
        z_seq = z_q.permute(0, 2, 1, 3).reshape(b, t, c * ch)  # (B, T_z, feature_dim)
        for layer in self.layers:
            z_seq = layer(z_seq)
        out = self.fc(z_seq)
        return out.view(b, t, c, ch).permute(0, 2, 1, 3)
#"""
