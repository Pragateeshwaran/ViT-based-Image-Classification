import torch
import torch.nn as nn
import torch.optim as optim
from model import ViT
from DataProcessing import train_loader, test_loader
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import time

# Ensure checkpoints directory exists
os.makedirs('checkpoints', exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ViT().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

log_file = r"checkpoints\training_log.txt"
with open(log_file, "w") as f:
    f.write("Epoch, Loss, Accuracy, Time\n")

def visualize_predictions(model, test_loader, device, num_images=6):
    model.eval()
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

    plt.figure(figsize=(12, 6))
    for i in range(num_images):
        plt.subplot(2, 3, i + 1)
        img = images[i].cpu().permute(1, 2, 0).numpy()
        plt.imshow(img)
        plt.title(f'Pred: {predicted[i].item()} | Label: {labels[i].item()}')
        plt.axis("off")
    plt.show()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = ViT().to(device)
# model.load_state_dict(torch.load("F:\works\A-important\A-neurals\ViT\checkpoints\model_epoch_60.pth"))
# model.eval() 
# print("Model loaded successfully.")
def train(model, train_loader, optimizer, criterion, device, epochs=500):
    best_loss = float('inf')
    patience, patience_limit = 0, 50

    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        total_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        scheduler.step()

        avg_loss = total_loss / len(train_loader)
        epoch_time = time.time() - start_time

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s")

        with open(log_file, "a") as f:
            f.write(f"{epoch+1}, {avg_loss:.4f}, {epoch_time:.2f}s\n")

        if (epoch + 1) % 10 == 0:
            visualize_predictions(model, test_loader, device)
            torch.save(model.state_dict(), f"checkpoints/afteroptim_model_epoch_{epoch+1}.pth")

        if avg_loss < best_loss:    
            best_loss = avg_loss
            patience = 0
            torch.save(model.state_dict(), "checkpoints/afteroptim_best_model.pth")
        else:
            patience += 1

        if patience >= patience_limit:
            print("Early stopping triggered.")
            break

def test(model, test_loader, device):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

train(model, train_loader, optimizer, criterion, device)
test(model, test_loader, device)
visualize_predictions(model, test_loader, device)
