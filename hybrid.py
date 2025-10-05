import torch
import torch.nn as nn

# Simple Mamba Block (pure PyTorch implementation, inspired by mamba-minimal)
class MambaBlock(nn.Module):
    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.dim = dim
        self.expand = expand
        self.d_inner = expand * dim
        self.d_conv = d_conv
        self.d_state = d_state
        
        # Input projection
        self.in_proj = nn.Linear(dim, self.d_inner * 2, bias=False)
        
        # Conv1d for local mixing
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv, 
                                groups=self.d_inner, bias=True, padding=d_conv-1)
        
        # Projections for SSM params
        self.x_proj = nn.Linear(self.d_inner, self.d_state * 2 + 1, bias=False)  # B, C, dt
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)  # Delta projection
        
        # A matrix (log for stability, shared across d_inner)
        ar = torch.arange(1, self.d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(ar))
        
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)

    def forward(self, x: torch.Tensor):
        # x: [B, seq_len, dim]
        b, seq_len, _ = x.shape
        
        # Project input
        x_and_res = self.in_proj(x)  # [B, seq_len, 2*d_inner]
        x, res = x_and_res.chunk(2, dim=-1)  # Split for skip connection
        
        # Reshape for conv (seq as channels for time-mixing)
        x = x.permute(0, 2, 1)  # [B, d_inner, seq_len]
        x = self.conv1d(x)[:, :, :seq_len]  # Conv and trim padding
        x = x.permute(0, 2, 1)  # Back to [B, seq_len, d_inner]
        x = torch.nn.functional.silu(x)  # Activation
        
        # SSM parameters from x
        ssm_params = self.x_proj(x)  # [B, seq_len, 2*d_state + 1]
        B, C, dt = torch.split(ssm_params, [self.d_state, self.d_state, 1], dim=-1)
        
        # Delta (dt) softplus for positivity
        dt = torch.nn.functional.softplus(self.dt_proj(dt))  # [B, seq_len, d_inner]
        
        # A matrix (discretize)
        A = -torch.exp(self.A_log)  # [d_state]
        
        # Simple SSM scan (sequential for simplicity)
        y = torch.zeros_like(x)
        h = torch.zeros(b, self.d_inner, self.d_state, device=x.device)  # Hidden state
        for t in range(seq_len):
            xt = x[:, t, :]  # [B, d_inner]
            dt_t = dt[:, t, :]  # [B, d_inner]
            B_t = B[:, t, :]  # [B, d_state]
            C_t = C[:, t, :]  # [B, d_state]
            
            # Discretize A and B
            A_disc = torch.exp(A.unsqueeze(0).unsqueeze(1) * dt_t.unsqueeze(-1))  # [B, d_inner, d_state]
            B_disc = dt_t.unsqueeze(-1) * B_t.unsqueeze(1)  # [B, d_inner, d_state]
            
            # State update
            h = A_disc * h + B_disc * xt.unsqueeze(-1)  # [B, d_inner, d_state]
            
            # Output
            y[:, t, :] = torch.einsum('bed,bd->be', h, C_t) + self.D * xt  # Cleaner with einsum
        
        # Residual and output proj
        y = y + res  # Skip connection
        y = torch.nn.functional.silu(y)
        y = self.out_proj(y)
        
        return y

# Hybrid Mamba-Transformer Refiner (unchanged)
class HybridMambaTransformerRefiner(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 128, num_layers: int = 2, num_heads: int = 4, d_state: int = 16, d_conv: int = 4):
        """
        Hybrid refiner: Alternates Mamba blocks for efficient local mixing and Transformer layers for global attention.
        - feature_dim: Input feature dim (num_filter * C_z)
        - hidden_dim: FFN dim for Transformer
        - num_layers: Total layers (half Mamba, half Transformer)
        - num_heads: For Transformer attention
        - d_state, d_conv: For Mamba
        """
        super().__init__()
        self.feature_dim = feature_dim
        
        # Positional embedding for Transformer
        self.pos_encoder = nn.Parameter(torch.randn(1, 1, feature_dim))  # Learnable
        
        # Hybrid layers: Alternate Mamba and Transformer
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i % 2 == 0:
                # Mamba layer
                self.layers.append(MambaBlock(dim=feature_dim, d_state=d_state, d_conv=d_conv))
            else:
                # Transformer layer
                self.layers.append(nn.TransformerEncoderLayer(d_model=feature_dim, nhead=num_heads, 
                                                            dim_feedforward=hidden_dim, dropout=0.1, batch_first=True))
        
        # Final projection back to feature_dim
        self.fc = nn.Linear(feature_dim, feature_dim)

    def forward(self, z_q: torch.Tensor):
        b, c, t, ch = z_q.shape
        z_seq = z_q.permute(0, 2, 1, 3).reshape(b, t, c * ch)  # [B, T_z, feature_dim]
        
        # Add positional encoding
        z_seq = z_seq + self.pos_encoder.expand(b, t, -1)
        
        # Pass through hybrid layers
        for layer in self.layers:
            z_seq = layer(z_seq)
        
        # Final projection
        out = self.fc(z_seq)  # [B, T_z, feature_dim]
        
        # Reshape back to original latent shape
        out = out.view(b, t, c, ch).permute(0, 2, 1, 3)  # [B, num_filter, T_z, C_z]
        return out
