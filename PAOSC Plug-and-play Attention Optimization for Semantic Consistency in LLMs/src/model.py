import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def get_sinusoid_encoding_table(max_len, d_model):
    def get_angle(pos, i):
        return pos / (10000 ** (2 * (i // 2) / d_model))
    
    table = torch.zeros(max_len, d_model)
    for pos in range(max_len):
        for i in range(0, d_model, 2):
            table[pos, i] = math.sin(get_angle(pos, i))
            if i + 1 < d_model:
                table[pos, i + 1] = math.cos(get_angle(pos, i))
    
    return table  # (max_len, d_model)

    
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, smooth_std=1.0, smooth_kernel_size=5):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.smooth_std = smooth_std
        self.smooth_kernel_size = smooth_kernel_size

    def smooth_attention(self, attn_weights):

        B, H, T, _ = attn_weights.shape
        device = attn_weights.device
        
        half_k = self.smooth_kernel_size // 2
        positions = torch.arange(-half_k, half_k + 1, dtype=torch.float32, device=device)
        kernel = torch.exp(-positions**2 / (2 * self.smooth_std**2))
        kernel = kernel / kernel.sum()  # normalize
        kernel = kernel.view(1, 1, -1)  # [1, 1, K]
    
        attn_weights = attn_weights.view(B * H * T, 1, T)
        smoothed = F.conv1d(attn_weights, kernel, padding=half_k)
    
        return smoothed.view(B, H, T, T)

    def forward(self, x, mask=None):
        B, T, C = x.size()

        # QKV projection
        qkv = self.qkv_proj(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # (B, T, H, D)
        q = q.transpose(1, 2)        # (B, H, T, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, H, T, T)
        if mask is not None:
            mask_exp = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
            attn_scores = attn_scores.masked_fill(mask_exp == 0, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, H, T, T)
        attn_weights = self.smooth_attention(attn_weights)  # (B, H, T, T

        attn_output = torch.matmul(attn_weights, v)    # (B, H, T, D)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)

        return self.out_proj(attn_output), attn_weights


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),  
            nn.Linear(4 * embed_dim, embed_dim),
        )
        self.ln2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out, attn_weights = self.attn(x, mask)
        x = self.ln1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.ln2(x + self.dropout(ffn_out))
        return x, attn_weights


class TransformerDiscriminator(nn.Module):
    def __init__(self, hidden_dim=4096, num_heads=4, num_layers=3, max_len=4096, dropout=0.1):
        super().__init__()
        self.embedding_proj = nn.Linear(hidden_dim, hidden_dim)

        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, dropout) for _ in range(num_layers)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)
        )

        sinusoid_table = get_sinusoid_encoding_table(max_len, hidden_dim)
        self.register_buffer('pos_embedding', sinusoid_table.unsqueeze(0))  # [1, max_len, hidden_dim]
        self.max_len = max_len

    def forward(self, x, mask=None):
        """
        x: [B, T, D]
        mask: [B, T] (float, 1 for real token, 0 for padding)
        """
        x = self.embedding_proj(x)  # [B, T, D]
        B, T, D = x.shape

        x = x + self.pos_embedding[:, :T, :]

        for block in self.blocks:
            x, _ = block(x, mask)

        # ----- Masked Mean Pooling -----
        if mask is not None:
            mask = mask.unsqueeze(-1)  # [B, T, 1]
            x = x * mask
            x_pooled = x.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-8)  # [B, D]
        else:
            x_pooled = x.mean(dim=1)

        logits = self.classifier(x_pooled)  # [B, 1]
        return logits


class TransformerGenerator(nn.Module):
    def __init__(self, embed_dim=4096, num_heads=4, num_layers=3, dropout=0.1, max_len=4096,temperature=1.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)
        ])

        sinusoid_table = get_sinusoid_encoding_table(max_len, embed_dim)
        self.register_buffer('pos_embedding', sinusoid_table.unsqueeze(0))  # [1, max_len, embed_dim]
        self.temperature = temperature

    def forward(self, x, mask=None):
        B, L, D = x.size()
        emb = x + self.pos_embedding[:, :L, :]
        # ----- Masked Mean Pooling -----
        all_attn = []
        
        if mask is not None:
            for block in self.blocks:
                hidden_states, attn_weights = block(emb,mask)
                all_attn.append(attn_weights)
        else:
            for block in self.blocks:
                hidden_states, attn_weights = block(emb)
                all_attn.append(attn_weights)

        return hidden_states, all_attn


class ClassificationHead(nn.Module):
    def __init__(self, embed_dim=4096, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.project = nn.Linear(embed_dim, 512)
        self.score_mlp = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)
        )

    def forward(self, x):  # x: [B, L, D]
        x = x.mean(dim=1)  # [B, D]
        x = self.norm(x)
        x = self.project(x)
        return self.score_mlp(x)
