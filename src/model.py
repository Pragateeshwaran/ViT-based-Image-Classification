import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

# ----------------- Data Preparation -----------------
# Improved augmentation and proper normalization for 3 channels
transform_train = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
transform_test = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Note: Adjust the root directories as needed.
train_dataset = ImageFolder(root=r'asset\Data\train', transform=transform_train)
test_dataset = ImageFolder(root=r'asset\Data\train', transform=transform_test)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# ----------------- Custom Self-Attention Layer -----------------
class CustomSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CustomSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "Embedding dim must be divisible by heads."

        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_o = nn.Linear(embed_dim, embed_dim, bias=False)

        self.scale = self.head_dim ** 0.5

    def forward(self, x):
        B, N, C = x.shape

        Q = self.W_q(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_probs = F.softmax(attn_scores, dim=-1)

        attn_output = torch.matmul(attn_probs, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, C)

        return self.W_o(attn_output)

# ----------------- Transformer Encoder Block with Dropout -----------------
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
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
        x = x + self.dropout1(self.attn(self.norm1(x)))
        x = x + self.mlp(self.norm2(x))
        return x

# ----------------- Patch Embedding Using Conv2d -----------------
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_channels=3, embed_dim=512):
        super(PatchEmbedding, self).__init__()
        self.num_patches = (img_size // patch_size) ** 2
        # Using Conv2d to extract patches and preserve spatial info
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.proj(x)  # [B, embed_dim, H/patch_size, W/patch_size]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        return x

# ----------------- SquamoScan Model -----------------
class SquamoScan(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_channels=3, embed_dim=512, 
                 depth=8, num_heads=8, num_classes=2, dropout=0.1):
        super(SquamoScan, self).__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        num_patches = self.patch_embed.num_patches
        # Learned positional embedding initialized with truncated normal
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.encoder = nn.Sequential(*[TransformerEncoder(embed_dim, num_heads, dropout=dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [B, num_patches+1, embed_dim]
        x = x + self.pos_embed
        x = self.encoder(x)
        x = self.norm(x)
        return self.head(x[:, 0])  # Use the cls token output

# ----------------- Training Setup -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SquamoScan().to(device)

# Using AdamW with weight decay and label smoothing for more stable training
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
num_epochs = 100

# Optional: CosineAnnealingLR scheduler to decay learning rate
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# ----------------- Training Loop -----------------
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# ----------------- Testing Loop -----------------
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")

# ----------------- Visualizing Predictions -----------------
def visualize_predictions():
    model.eval()
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs, dim=1)
    fig, axes = plt.subplots(1, 6, figsize=(15, 3))
    for i in range(6):
        img = images[i].cpu().permute(1, 2, 0)
        img = (img * 0.5 + 0.5).clamp(0, 1)
        axes[i].imshow(img)
        axes[i].set_title(f"Pred: {predicted[i].item()}, Label: {labels[i].item()}")
        axes[i].axis("off")
    plt.show()

visualize_predictions()
