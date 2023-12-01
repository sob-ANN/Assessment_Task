import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RotaryPositionEmbedding(nn.Module):
    def __init__(self, d_model, dim_head, n_ctx):
        super().__init__()
        self.d_model = d_model
        self.dim_head = dim_head
        self.n_ctx = n_ctx

        position = torch.arange(0, n_ctx)
        num_timescales = d_model // dim_head
        inv_timescales = 1 / torch.exp(
            torch.arange(0, num_timescales, dtype=torch.float) *
            (-math.log(10000) / (num_timescales - 1))
        )
        scaled_time = position * inv_timescales
        signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)
        self.register_buffer('position', signal)

    def forward(self, x):
        # Apply rotary positional embedding
        x = x * self.position.unsqueeze(0)
        return x
