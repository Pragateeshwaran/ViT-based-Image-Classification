import torch
import torch.nn as nn
import torch.nn.functional as F

# Custom Self-Attention Layer for multi-head attention mechanism
class CustomSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        """Initialize self-attention layer parameters.
        
        Args:
            embed_dim (int): Embedding dimension
            num_heads (int): Number of attention heads
        """
        super(CustomSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"

        # Linear transformations for query, key, value, and output
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_o = nn.Linear(embed_dim, embed_dim, bias=False)

        self.scale = self.head_dim ** 0.5  # Scaling factor for dot product attention

    def forward(self, x):
        """Forward pass of self-attention layer.
        
        Args:
            x (torch.Tensor): Input tensor [batch_size, seq_len, embed_dim]
        Returns:
            torch.Tensor: Output tensor after attention
        """
        B, N, C = x.shape  # Batch size, sequence length, embedding dimension

        # Transform and reshape for multi-head attention
        Q = self.W_q(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Calculate attention scores and probabilities
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_probs = F.softmax(attn_scores, dim=-1)

        # Apply attention to values and reshape
        attn_output = torch.matmul(attn_probs, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, C)

        return self.W_o(attn_output)

# Transformer Encoder Block with dropout for regularization
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        """Initialize transformer encoder block.
        
        Args:
            embed_dim (int): Embedding dimension
            num_heads (int): Number of attention heads
            mlp_ratio (float): Ratio for MLP hidden dimension
            dropout (float): Dropout probability
        """
        super(TransformerEncoder, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = CustomSelfAttention(embed_dim, num_heads)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """Forward pass of transformer encoder.
        
        Args:
            x (torch.Tensor): Input tensor
        Returns:
            torch.Tensor: Output tensor after transformation
        """
        x = x + self.dropout1(self.attn(self.norm1(x)))  # Residual connection with attention
        x = x + self.mlp(self.norm2(x))  # Residual connection with MLP
        return x

# Patch Embedding layer to convert images into patch sequences
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_channels=3, embed_dim=512):
        """Initialize patch embedding layer.
        
        Args:
            img_size (int): Input image size
            patch_size (int): Size of each patch
            in_channels (int): Number of input channels
            embed_dim (int): Embedding dimension
        """
        super(PatchEmbedding, self).__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """Convert images to patch embeddings.
        
        Args:
            x (torch.Tensor): Input images [B, C, H, W]
        Returns:
            torch.Tensor: Patch embeddings [B, num_patches, embed_dim]
        """
        x = self.proj(x)  # Extract patches
        x = x.flatten(2).transpose(1, 2)  # Flatten and transpose to sequence
        return x

# Vision Transformer (ViT) Model for image classification
class ViT(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_channels=3, embed_dim=512, 
                 depth=8, num_heads=8, num_classes=2, dropout=0.1):
        """Initialize Vision Transformer model.
        
        Args:
            img_size (int): Input image size
            patch_size (int): Size of each patch
            in_channels (int): Number of input channels
            embed_dim (int): Embedding dimension
            depth (int): Number of transformer blocks
            num_heads (int): Number of attention heads
            num_classes (int): Number of output classes
            dropout (float): Dropout probability
        """
        super(ViT, self).__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        num_patches = self.patch_embed.num_patches
        
        # Positional embedding with truncated normal initialization
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Stack of transformer encoder blocks
        self.encoder = nn.Sequential(*[TransformerEncoder(embed_dim, num_heads, dropout=dropout) 
                                     for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        """Forward pass of ViT model.
        
        Args:
            x (torch.Tensor): Input images [B, C, H, W]
        Returns:
            torch.Tensor: Class logits [B, num_classes]
        """
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.encoder(x)
        x = self.norm(x)
        return self.head(x[:, 0])  # Output from cls token