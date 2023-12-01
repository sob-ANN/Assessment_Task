import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupQueryAttention(nn.Module):
    def __init__(self, n_heads, n_emb, n_ctx, num_groups=8, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.n_emb = n_emb
        self.n_ctx = n_ctx
        self.num_groups = num_groups

        self.attn = nn.MultiheadAttention(n_emb, n_heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        # Apply group query attention
        q_groups = torch.split(q, self.num_groups, dim=0)
        k_groups = torch.split(k, self.num_groups, dim=0)
        v_groups = torch.split(v, self.num_groups, dim=0)

        attn_weights = []
        for q_group, k_group, v_group in zip(q_groups, k_groups, v_groups):
            x, attn_weight = self.attn(q_group, k_group, v_group)
            attn_weights.append(attn_weight)

        attn_weights = torch.cat(attn_weights, dim=0)
        x = torch.cat(x, dim=0)
        x = self.dropout(x)
        return x, attn_weights
