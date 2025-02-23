import torch
from torch import nn
import torch.nn.functional as F
from model import ViT
from DataProcessing import train_loader, test_loader
import matplotlib.pyplot as plt
from tqdm import tqdm

# Training configuration and setup
def train_model():
    """Main training function for ViT model."""
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViT().to(device)

    # Optimizer and loss function setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    num_epochs = 100
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Training loop
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

    # Testing loop
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
    return model

# Visualization of predictions
def visualize_predictions(model):
    """Visualize model predictions on test samples.
    
    Args:
        model (nn.Module): Trained ViT model
    """
    model.eval()
    images, labels = next(iter(test_loader))
    device = next(model.parameters()).device
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs, dim=1)
    
    # Create visualization
    fig, axes = plt.subplots(1, 6, figsize=(15, 3))
    for i in range(6):
        img = images[i].cpu().permute(1, 2, 0)
        img = (img * 0.5 + 0.5).clamp(0, 1)  # Denormalize
        axes[i].imshow(img)
        axes[i].set_title(f"Pred: {predicted[i].item()}, Label: {labels[i].item()}")
        axes[i].axis("off")
    plt.show()

if __name__ == "__main__":
    trained_model = train_model()
    visualize_predictions(trained_model)