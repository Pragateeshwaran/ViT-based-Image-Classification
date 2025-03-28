import torch
import torch.nn as nn
import torch.nn.functional as F

# Self-Attention Mechanism
class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"

        # Linear layers for queries, keys, values, and output projection (no bias)
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_o = nn.Linear(embed_dim, embed_dim, bias=False)

        # Scaling factor for dot-product attention
        self.scale = self.head_dim ** 0.5

    def forward(self, x):
        B, N, C = x.shape  # [batch_size, num_tokens, embed_dim]

        # Compute Q, K, V and reshape for multi-head attention
        Q = self.W_q(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, N, head_dim]
        K = self.W_k(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores and probabilities
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, num_heads, N, N]
        attn_probs = F.softmax(attn_scores, dim=-1)

        # Attention output
        attn_output = torch.matmul(attn_probs, V)  # [B, num_heads, N, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, C)  # [B, N, embed_dim]

        return self.W_o(attn_output)

# Transformer Encoder Block
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        # Pre-norm architecture (LayerNorm before attention and MLP)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = SelfAttention(embed_dim, num_heads)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        # MLP with expansion factor (mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Residual connection around attention
        x = x + self.dropout1(self.attn(self.norm1(x)))
        # Residual connection around MLP
        x = x + self.mlp(self.norm2(x))
        return x

# Patch Embedding (Manual Patch Extraction)
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_channels=3, embed_dim=512):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2

        # Linear projection for flattened patches
        self.proj = nn.Linear(patch_size * patch_size * in_channels, embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape  # [batch_size, channels, height, width]
        p = self.patch_size

        # Reshape image into patches: [B, C, H//p, p, W//p, p]
        x = x.view(B, C, H // p, p, W // p, p)
        # Rearrange to [B, num_patches, C*p*p]: permute to [B, H//p, W//p, C, p, p], then flatten patches
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(B, self.num_patches, C * p * p)
        # Project each patch to embed_dim
        x = self.proj(x)  # [B, num_patches, embed_dim]
        return x

# Vision Transformer (ViT)
class ViT(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_channels=3, embed_dim=512, 
                 depth=8, num_heads=8, num_classes=2, dropout=0.1):
        super(ViT, self).__init__()
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # Class token (learnable parameter)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional embeddings for cls token + patch tokens
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)  # Initialize with truncated normal

        # Transformer encoder layers
        self.encoder = nn.Sequential(*[
            TransformerEncoder(embed_dim, num_heads, dropout=dropout)
            for _ in range(depth)
        ])

        # Final layer norm and classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]  # Batch size

        # Extract patch embeddings
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]

        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        x = torch.cat([cls_tokens, x], dim=1)  # [B, num_patches + 1, embed_dim]

        # Add positional embeddings
        x = x + self.pos_embed  # [B, num_patches + 1, embed_dim]

        # Pass through transformer encoder
        x = self.encoder(x)  # [B, num_patches + 1, embed_dim]

        # Apply layer norm
        x = self.norm(x)

        # Extract cls token output and classify
        cls_output = x[:, 0]  # [B, embed_dim]
        return self.head(cls_output)  # [B, num_classes]

# Example usage
if __name__ == "__main__":
    # Create a sample input (batch_size=2, channels=3, height=256, width=256)
    x = torch.randn(2, 3, 256, 256)
    model = ViT(
        img_size=256,
        patch_size=16,
        in_channels=3,
        embed_dim=512,
        depth=8,
        num_heads=8,
        num_classes=2,
        dropout=0.1
    )
    output = model(x)
    print(f"Output shape: {output.shape}")  # Should be [2, 2]