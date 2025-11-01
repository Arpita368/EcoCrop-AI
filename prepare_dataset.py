import os
import shutil
import random

# Base paths
base_dir = "dataset/plantvillage"
train_dir = "dataset/train"
valid_dir = "dataset/valid"

# Create target directories if not exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(valid_dir, exist_ok=True)

# Split ratio
split_ratio = 0.8  # 80% train, 20% validation

# Loop through each class folder
for class_name in os.listdir(base_dir):
    class_path = os.path.join(base_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    # Create class subfolders in train and valid
    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(valid_dir, class_name), exist_ok=True)

    # List all images in this class
    images = os.listdir(class_path)
    random.shuffle(images)

    split_point = int(len(images) * split_ratio)
    train_images = images[:split_point]
    valid_images = images[split_point:]

    # Copy images
    for img in train_images:
        shutil.copy(os.path.join(class_path, img), os.path.join(train_dir, class_name, img))
    for img in valid_images:
        shutil.copy(os.path.join(class_path, img), os.path.join(valid_dir, class_name, img))

    print(f"✅ {class_name}: {len(train_images)} train, {len(valid_images)} valid")

print("\n✅ Dataset split complete! Check 'dataset/train' and 'dataset/valid' folders.")
