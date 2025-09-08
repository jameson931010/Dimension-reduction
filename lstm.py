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
