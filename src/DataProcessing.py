import os
import shutil
import random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def prepare_test_data(data_dir=r"asset/Data/train", test_dir=r"asset/Data/test", split_ratio=0.2):
    """Creates test data by moving a portion of train data if /test does not exist."""
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
        
        for class_folder in os.listdir(data_dir):
            class_path = os.path.join(data_dir, class_folder)
            if os.path.isdir(class_path):
                os.makedirs(os.path.join(test_dir, class_folder), exist_ok=True)

                images = [img for img in os.listdir(class_path) if img.endswith(('.jpg', '.png'))]
                random.shuffle(images)

                num_test = int(len(images) * split_ratio)
                test_images = images[:num_test]

                for img in test_images:
                    src = os.path.join(class_path, img)
                    dest = os.path.join(test_dir, class_folder, img)
                    shutil.move(src, dest)
        print("✅ Test data prepared successfully!")
    else:
        print("✅ Test data already exists.")

# Image transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Prepare test data if not exists
if __name__ == "__main__":
    prepare_test_data()
    print("Data Processing completed successfully!")

# Load data
dataset_train = datasets.ImageFolder(r"asset/Data/train", transform=transform)
dataset_test = datasets.ImageFolder(r"asset/Data/test", transform=transform)

train_loader = DataLoader(dataset_train, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset_test, batch_size=64, shuffle=True)

print(f"the train data has {len(dataset_train)} images")
print(f"the test data has {len(dataset_test)} images")
