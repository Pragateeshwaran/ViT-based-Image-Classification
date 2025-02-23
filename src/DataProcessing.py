import os
import cv2
import numpy as np
from tqdm import tqdm
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Data augmentation and preprocessing for training
transform_train = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Preprocessing for testing
transform_test = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load datasets
train_dataset = ImageFolder(root=r'asset\Data\train', transform=transform_train)
test_dataset = ImageFolder(root=r'asset\Data\test', transform=transform_test)

# Create data loaders
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

def resize_with_padding(image, target_size=(256, 256), pad_color=(0, 0, 0)):
    """Resize image with padding to maintain aspect ratio.
    
    Args:
        image (np.ndarray): Input image
        target_size (tuple): Desired output size
        pad_color (tuple): Color for padding
    Returns:
        np.ndarray: Resized and padded image
    """
    h, w = image.shape[:2]
    scale = min(target_size[0] / h, target_size[1] / w)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    delta_w, delta_h = target_size[1] - new_w, target_size[0] - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    return cv2.copyMakeBorder(resized, top, bottom, left, right, 
                            cv2.BORDER_CONSTANT, value=pad_color)

def resize_images():
    """Process and resize all images in dataset."""
    dataset_path = "./data"
    folders = ["train", "test"]

    for folder in folders:
        folder_path = os.path.join(dataset_path, folder)
        for subfolder in os.listdir(folder_path):
            subfolder_path = os.path.join(folder_path, subfolder)
            if not os.path.isdir(subfolder_path):
                continue

            print(f"Processing: {subfolder_path}")
            for img_name in tqdm(os.listdir(subfolder_path)):
                img_path = os.path.join(subfolder_path, img_name)
                if not img_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
                    continue

                image = cv2.imread(img_path)
                if image is None:
                    print(f"Skipping corrupted image: {img_path}")
                    continue

                resized_image = resize_with_padding(image)
                cv2.imwrite(img_path, resized_image)

    print("✅ All images resized successfully!")

if __name__ == '__main__':
    resize_images()