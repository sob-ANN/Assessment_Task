import torch
import torch.nn as nn
import torch.nn.functional as F

class GPT2Attention(nn.Module):
    def __init__(self, n_heads, n_emb, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = n_emb // n_heads
        
        # Q, K, V weight matrices for each head
        self.Wq = nn.Linear(n_emb, n_emb)
        self.Wk = nn.Linear(n_emb, n_emb)
        self.Wv = nn.Linear(n_emb, n_emb)
        
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(n_emb, n_emb)
        
    def forward(self, x):
        # Split input into multiple heads
        q = self.Wq(x).view(x.size(0), -1, self.n_heads, self.d_k).transpose(1, 2) # BxHxLxD_k
        k = self.Wk(x).view(x.size(0), -1, self.n_heads, self.d_k).transpose(1, 2) # BxHxLxD_k
        v = self.Wv(x).view(x.size(0), -1, self.n_heads, self.d_k).transpose(1, 2) # BxHxLxD_k
        
        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5) # BxHxLxL
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(self.dropout(attn_weights), v) # BxHxLxD_k
        
        # Merge heads and perform linear transformation
        attn_output = attn_output.transpose(1, 2).contiguous().view(x.size(0), -1, self.n_heads * self.d_k) # BxLxH*D_k
        return self.out(attn_output)

class GPT2PositionalEncoding(nn.Module):
    def __init__(self, n_emb, max_len=512):
        super().__init__()
        self.pos_enc = self.get_positional_encoding(n_emb, max_len)
        
    def get_positional_encoding(self, n_emb, max_len):
        pos = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_emb, 2) * (-torch.log(torch.tensor(10000.0)) / n_emb))
        pos_enc = torch.zeros(max_len, n_emb)
        pos_enc[:, 0::2] = torch.sin(pos * div_term)
        pos_enc[:, 1::2] = torch.cos(pos * div_term)
        pos_enc = pos_enc.unsqueeze(0)
        return pos_enc
        
    def forward(self, x):
        return x + self.pos_enc[:, :x.size(1), :]

class GPT2Layer(nn.Module):
    def __init__(self, n_heads, n_emb, dropout=0.1):
        super().__init__()
        self.attn = GPT2Attention(n_heads, n_emb, dropout)
        self.norm1 = nn.LayerNorm(n_emb)
        
    def forward(self, x):
        # Apply attention mechanism with layer normalization
        x = x + self.attn(self.norm1(x))
        return x

class GPT2(nn.Module):
    def __init__(self, n_vocab, n_embd, n_layers, n_heads, d_ff=2048, max_len=512, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(n_vocab, n_embd)
        self.pos_enc = GPT2PositionalEncoding(n_embd, max_len)
        
        self.transformer_layers = nn.ModuleList([GPT2Layer(n_heads, n_embd, dropout) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.to_logits = nn.Linear(n_embd, n_vocab)

    def forward(self, x):
        # Embed tokens and add positional encoding
        token_embeddings = self.token_emb(x)
        x = self.pos_enc(token_embeddings)
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x)
        
        # Apply dropout and output logits
        x = self.dropout(x)
        x = self.to_logits(x)
        return x
