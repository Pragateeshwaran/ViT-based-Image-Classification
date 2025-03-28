import torch
from torch import nn
import torch.nn.functional as F
from model import ViT
from DataProcessing import train_loader, test_loader
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

# Evolutionary Strategy Configuration
POPULATION_SIZE = 20
SIGMA = 0.1  # Noise standard deviation
ALPHA = 0.03  # Learning rate for ES

# ES Training Function
def train_model_es():
    """Main training function using Evolutionary Strategies for ViT model."""
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViT().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Population setup
    params = model.state_dict()
    param_vector = torch.nn.utils.parameters_to_vector(model.parameters()).detach()

    for epoch in range(100):
        # Generate noise for population
        noise_population = [torch.randn_like(param_vector) * SIGMA for _ in range(POPULATION_SIZE)]

        # Evaluate each individual in the population
        rewards = []
        for noise in noise_population:
            new_params = param_vector + noise
            torch.nn.utils.vector_to_parameters(new_params, model.parameters())
            model.eval()
            
            total_loss = 0
            with torch.no_grad():
                for images, labels in train_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    total_loss += loss.item()

            rewards.append(-total_loss)  # Negative loss as reward

        # Update parameters using weighted combination of population
        weighted_sum = sum(reward * noise for reward, noise in zip(rewards, noise_population))
        param_vector += (ALPHA / (POPULATION_SIZE * SIGMA)) * weighted_sum
        torch.nn.utils.vector_to_parameters(param_vector, model.parameters())

        avg_loss = -sum(rewards) / POPULATION_SIZE
        print(f"Epoch [{epoch+1}/100], Avg Loss: {avg_loss:.4f}")

    return model

# Visualization of predictions
def visualize_predictions(model):
    """Visualize model predictions on test samples."""
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
    trained_model = train_model_es()
    visualize_predictions(trained_model)
