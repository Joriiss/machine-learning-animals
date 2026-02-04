import os
import numpy as np
from tensorflow import keras
import cv2
from sklearn.metrics import confusion_matrix, recall_score

# Configuration
IMG_SIZE = 128
MODEL_PATH = 'models/animal_classifier.h5'

# Class mapping (first letter to class name)
CLASS_MAP = {
    'e': 'elephant',
    't': 'tigre',
    'g': 'giraffe'
}

CLASSES = ["elephant", "tigre", "giraffe"]

def load_test_images(test_dir):
    """Load test images and their expected labels from filenames"""
    images = []
    expected_labels = []
    filenames = []
    
    if not os.path.exists(test_dir):
        print(f"Error: {test_dir} does not exist!")
        return images, expected_labels, filenames
    
    for filename in os.listdir(test_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(test_dir, filename)
            try:
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    images.append(img)
                    
                    # Get expected label from first letter of filename
                    first_letter = filename[0].lower()
                    if first_letter in CLASS_MAP:
                        expected_labels.append(CLASSES.index(CLASS_MAP[first_letter]))
                    else:
                        print(f"Warning: Unknown class for {filename} (first letter: {first_letter})")
                        expected_labels.append(-1)  # Unknown
                    
                    filenames.append(filename)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
    
    return images, expected_labels, filenames

# Load the model
print(f"Loading model from {MODEL_PATH}...")
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file {MODEL_PATH} not found! Please train the model first.")
    exit(1)

model = keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

# Load test images
print("\nLoading test images...")
test_images, expected_labels, filenames = load_test_images('data/test')

if len(test_images) == 0:
    print("No test images found!")
    exit(1)

print(f"Loaded {len(test_images)} test images")

# Preprocess images
X_test = np.array(test_images) / 255.0

# Make predictions
print("\nMaking predictions...")
predictions = model.predict(X_test, verbose=0)
predicted_classes = np.argmax(predictions, axis=1)

# Filter out unknown classes (-1) for metrics calculation
valid_indices = [i for i in range(len(expected_labels)) if expected_labels[i] != -1]
y_true = np.array([expected_labels[i] for i in valid_indices])
y_pred = np.array([predicted_classes[i] for i in valid_indices])

# Calculate accuracy
correct = 0
total = 0

# Track accuracy by class
class_correct = {i: 0 for i in range(len(CLASSES))}
class_total = {i: 0 for i in range(len(CLASSES))}

print("\n" + "="*60)
print("Test Results:")
print("="*60)

for i in range(len(test_images)):
    expected = expected_labels[i]
    predicted = predicted_classes[i]
    confidence = predictions[i][predicted] * 100
    
    if expected == -1:
        status = "UNKNOWN CLASS"
    elif expected == predicted:
        status = "✓ CORRECT"
        correct += 1
        if expected in class_correct:
            class_correct[expected] += 1
    else:
        status = "✗ WRONG"
    
    total += 1
    
    # Count total per class
    if expected != -1 and expected in class_total:
        class_total[expected] += 1
    
    print(f"\n{filenames[i]}")
    print(f"  Expected: {CLASSES[expected] if expected != -1 else 'Unknown'}")
    print(f"  Predicted: {CLASSES[predicted]} ({confidence:.1f}%)")
    print(f"  Status: {status}")

print("\n" + "="*60)
if total > 0:
    # Overall accuracy
    accuracy = (correct / total) * 100
    print(f"\nTaux de précision global: {correct}/{total} = {accuracy:.1f}%")
    
    # Calculate recall per class
    if len(y_true) > 0:
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        
        print("\nRappel par classe:")
        print("-" * 60)
        for i, class_name in enumerate(CLASSES):
            if class_total[i] > 0:
                recall = recall_per_class[i] * 100
                print(f"  {class_name.capitalize()}: {recall:.1f}%")
            else:
                print(f"  {class_name.capitalize()}: No test images")
        
        # Confusion matrix
        print("\nMatrice de confusion:")
        print("-" * 60)
        cm = confusion_matrix(y_true, y_pred)
        
        # Print header
        print(" " * 12, end="")
        for class_name in CLASSES:
            print(f"{class_name.capitalize():>12}", end="")
        print()
        
        # Print matrix
        for i, class_name in enumerate(CLASSES):
            print(f"{class_name.capitalize():>12}", end="")
            for j in range(len(CLASSES)):
                print(f"{cm[i][j]:>12}", end="")
            print()
        
        # Print accuracy by class (for reference)
        print("\nPrécision par classe:")
        print("-" * 60)
        for i, class_name in enumerate(CLASSES):
            if class_total[i] > 0:
                class_accuracy = (class_correct[i] / class_total[i]) * 100
                print(f"  {class_name.capitalize()}: {class_correct[i]}/{class_total[i]} = {class_accuracy:.1f}%")
            else:
                print(f"  {class_name.capitalize()}: No test images")
else:
    print("\nNo valid test images found!")
print("="*60)
