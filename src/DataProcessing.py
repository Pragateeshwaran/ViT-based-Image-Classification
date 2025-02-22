import os
import cv2
import numpy as np
from tqdm import tqdm  # Progress bar

# Resize function
def resize_with_padding(image, target_size=(256, 256), pad_color=(0, 0, 0)):
    h, w = image.shape[:2]
    scale = min(target_size[0] / h, target_size[1] / w)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    delta_w, delta_h = target_size[1] - new_w, target_size[0] - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color)
    return padded

dataset_path = "asset/Data"  # Update this if needed
folders = ["train", "test", "val"]

# Iterate over train, test, and val folders
for folder in folders:
    folder_path = os.path.join(dataset_path, folder)

    # Process subfolders (class folders)
    for subfolder in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder)
        
        if not os.path.isdir(subfolder_path):  # Skip files, only process folders
            continue

        print(f"Processing: {subfolder_path}")

        # Process each image inside the subfolder
        for img_name in tqdm(os.listdir(subfolder_path)):
            img_path = os.path.join(subfolder_path, img_name)

            if not img_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
                continue

            image = cv2.imread(img_path)
            if image is None:
                print(f"Skipping corrupted image: {img_path}")
                continue

            resized_image = resize_with_padding(image, target_size=(256, 256))

            # Overwrite the original image
            cv2.imwrite(img_path, resized_image)

print("✅ All images resized successfully!")
