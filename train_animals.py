import os
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import cv2

# Configuration
IMG_SIZE = 128  # Simple and fast
BATCH_SIZE = 32
EPOCHS = 10

# Data directories
DATA_DIR = "data"
CLASSES = ["elephant", "tigre", "giraffe"]

def load_images_from_folder(folder_path, label):
    """Load images from a folder and assign labels"""
    images = []
    labels = []
    
    if not os.path.exists(folder_path):
        print(f"Warning: {folder_path} does not exist. Skipping...")
        return images, labels
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            try:
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    images.append(img)
                    labels.append(label)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
    
    return images, labels

# Load all images
print("Loading images...")
all_images = []
all_labels = []

for idx, class_name in enumerate(CLASSES):
    folder_path = os.path.join(DATA_DIR, class_name)
    print(f"Loading {class_name} from {folder_path}...")
    images, labels = load_images_from_folder(folder_path, idx)
    all_images.extend(images)
    all_labels.extend(labels)
    print(f"Loaded {len(images)} {class_name} images")

# Convert to numpy arrays
X = np.array(all_images) / 255.0  # Normalize to [0, 1]
y = np.array(all_labels)

print(f"\nTotal images loaded: {len(X)}")
print(f"Image shape: {X[0].shape}")

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Convert labels to categorical
y_train = keras.utils.to_categorical(y_train, num_classes=len(CLASSES))
y_test = keras.utils.to_categorical(y_test, num_classes=len(CLASSES))

# Create a simple CNN model
print("\nCreating model...")
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D(2, 2),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(CLASSES), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(model.summary())

# Train the model
print("\nTraining model...")
history = model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_test, y_test),
    verbose=1
)

# Evaluate the model
print("\nEvaluating model...")
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest accuracy: {test_accuracy:.2%}")

# Save the model
model.save('animal_classifier.h5')
print("\nModel saved as 'animal_classifier.h5'")
