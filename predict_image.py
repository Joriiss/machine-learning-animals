import os
import numpy as np
from tensorflow import keras
import cv2
import sys

# Configuration
IMG_SIZE = 128
MODEL_PATH = 'models/animal_classifier.h5'

CLASSES = ["elephant", "tigre", "giraffe"]

def load_and_preprocess_image(image_path):
    """Load and preprocess a single image"""
    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} does not exist!")
        return None
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return None
    
    # Preprocess: resize, convert color, normalize
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0  # Normalize to [0, 1]
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img

# Load the model
print(f"Loading model from {MODEL_PATH}...")
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file {MODEL_PATH} not found! Please train the model first.")
    sys.exit(1)

model = keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

# Get image path
img_dir = 'data/img'
if not os.path.exists(img_dir):
    print(f"Error: Directory {img_dir} does not exist!")
    sys.exit(1)

# Get all images in the directory
image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

if len(image_files) == 0:
    print(f"Error: No images found in {img_dir}!")
    sys.exit(1)

# Store all results
results = []

# Process each image
for image_file in image_files:
    image_path = os.path.join(img_dir, image_file)
    
    # Load and preprocess image
    img = load_and_preprocess_image(image_path)
    if img is None:
        continue
    
    # Make prediction
    predictions = model.predict(img, verbose=0)
    probabilities = predictions[0]
    
    # Get predicted class
    predicted_class_idx = np.argmax(probabilities)
    predicted_class = CLASSES[predicted_class_idx]
    confidence = probabilities[predicted_class_idx] * 100
    
    # Calculate balance score (standard deviation - lower = more balanced)
    # Perfect balance would be [33.33, 33.33, 33.33] with std = 0
    balance_score = np.std(probabilities)
    
    # Store result
    results.append({
        'filename': image_file,
        'probabilities': probabilities,
        'predicted_class': predicted_class,
        'confidence': confidence,
        'balance_score': balance_score
    })

# Sort by balance score (least balanced first, most balanced last)
results.sort(key=lambda x: x['balance_score'], reverse=True)

# Display results sorted by balance
for result in results:
    print(f"\n{'='*60}")
    print(f"Processing: {result['filename']}")
    print('='*60)
    
    probabilities = result['probabilities']
    predicted_class = result['predicted_class']
    confidence = result['confidence']
    balance_score = result['balance_score']
    
    # Display results
    print(f"\nPrédiction: {predicted_class.capitalize()} ({confidence:.1f}%)")
    print(f"Score d'équilibre: {balance_score:.4f} (plus bas = plus équilibré)\n")
    print("Probabilités pour chaque classe:")
    print("-" * 60)
    
    # Sort by probability (highest first)
    sorted_indices = np.argsort(probabilities)[::-1]
    
    for idx in sorted_indices:
        class_name = CLASSES[idx]
        prob = probabilities[idx] * 100
        bar_length = int(prob / 2)  # Bar length proportional to probability
        bar = "█" * bar_length
        print(f"  {class_name.capitalize():10s}: {prob:5.1f}% {bar}")
