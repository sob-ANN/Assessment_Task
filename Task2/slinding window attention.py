import torch
import torch.nn as nn
import torch.nn.functional as F

class SlidingWindowAttention(nn.Module):
    def __init__(self, n_heads, n_emb, window_size, stride=1, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.n_emb = n_emb
        self.window_size = window_size
        self.stride = stride

        self.attn = nn.MultiheadAttention(n_emb, n_heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        # Apply sliding window attention
        q = q.unfold(-1, self.window_size, self.stride)
        k = k.unfold(-1, self.window_size, self.stride)
        v = v.unfold(-1, self.window_size, self.stride)

        x, attn_weights = self.attn(q, k, v)
        x = x.reshape(q.shape)
        attn_weights = attn_weights.reshape(attn_weights.shape[:-1])
        x = self.dropout(x)
        return x, attn_weights
