import os
import random
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# ===============================================
# 🗂️ 1️⃣ AUTO SPLIT DATASET (if not already split)
# ===============================================
base_dir = "dataset/plantvillage"
train_dir = "dataset/train"
valid_dir = "dataset/valid"

def split_dataset():
    """Splits dataset/plantvillage into train/ and valid/ automatically (80/20)."""
    if not os.path.exists(base_dir):
        raise Exception("❌ dataset/plantvillage not found. Please check your dataset path.")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)

    split_ratio = 0.8  # 80% train, 20% valid

    for class_name in os.listdir(base_dir):
        class_path = os.path.join(base_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        train_class_dir = os.path.join(train_dir, class_name)
        valid_class_dir = os.path.join(valid_dir, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(valid_class_dir, exist_ok=True)

        images = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
        random.shuffle(images)
        split_point = int(len(images) * split_ratio)
        train_images = images[:split_point]
        valid_images = images[split_point:]

        for img in train_images:
            shutil.copy(os.path.join(class_path, img), os.path.join(train_class_dir, img))
        for img in valid_images:
            shutil.copy(os.path.join(class_path, img), os.path.join(valid_class_dir, img))

        print(f"✅ {class_name}: {len(train_images)} train, {len(valid_images)} valid")

    print("\n✅ Dataset split complete! Check 'dataset/train' and 'dataset/valid' folders.")

# Only split if needed
if not os.path.exists(train_dir) or not os.path.exists(valid_dir):
    print("⚙️ Train/Valid folders not found — splitting dataset automatically...")
    split_dataset()
else:
    print("✅ Train/Valid folders already exist. Skipping split.")

# ===============================================
# 🧠 2️⃣ DATA PREPARATION
# ===============================================
train_gen = ImageDataGenerator(
    rescale=1/255.0,
    rotation_range=15,
    zoom_range=0.1,
    horizontal_flip=True
)
val_gen = ImageDataGenerator(rescale=1/255.0)

train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=(64, 64),
    batch_size=16,
    class_mode='categorical'
)

val_data = val_gen.flow_from_directory(
    valid_dir,
    target_size=(64, 64),
    batch_size=16,
    class_mode='categorical'
)

# ===============================================
# 🧩 3️⃣ MODEL CREATION (CNN)
# ===============================================
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(train_data.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# ===============================================
# 🎯 4️⃣ TRAINING CONFIGURATION
# ===============================================
checkpoint = ModelCheckpoint(
    "plant_disease_model.h5",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

# ===============================================
# 🚀 5️⃣ TRAIN THE MODEL
# ===============================================
EPOCHS = 8  # You can increase to 12–15 for better accuracy
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stop]
)

print("\n✅ Training complete! Model saved as 'plant_disease_model.h5'")

# ===============================================
# 📊 6️⃣ OPTIONAL: SAVE CLASS LABELS
# ===============================================
class_labels = list(train_data.class_indices.keys())
with open("class_labels.txt", "w") as f:
    for label in class_labels:
        f.write(f"{label}\n")
print("✅ Class labels saved in 'class_labels.txt'")
