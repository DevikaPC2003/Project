import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# === Step 1: Load image data ===
data_dir = 'dataset'
class_names = sorted(os.listdir(data_dir))  # ['class 1', 'class 2', 'class 3']
data = []
labels = []

print("🔍 Loading images...")
for class_name in class_names:
    class_path = os.path.join(data_dir, class_name)
    for file in os.listdir(class_path):
        file_path = os.path.join(class_path, file)
        img = cv2.imread(file_path)
        if img is not None:
            img = cv2.resize(img, (64, 64))  # Resize all images to 64x64
            data.append(img)
            labels.append(class_name)

print(f"✅ Loaded {len(data)} images from {len(class_names)} classes.")

# === Step 2: Preprocess data ===
data = np.array(data) / 255.0  # Normalize
labels = np.array(labels)

le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
labels_categorical = to_categorical(labels_encoded)

X_train, X_test, y_train, y_test = train_test_split(
    data, labels_categorical, test_size=0.2, random_state=42)

# === Step 3: Build CNN model ===
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(len(class_names), activation='softmax')  # 3 output classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# === Step 4: Train model ===
print("🚀 Training model...")
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# === Step 5: Save model ===
os.makedirs('model', exist_ok=True)
model.save('model/image_model.h5')
print("🎉 Model trained and saved to model/image_model.h5")
