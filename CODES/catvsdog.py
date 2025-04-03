import os
import cv2
import numpy as np
import zipfile
import urllib.request
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from skimage.feature import hog

# ---------------------------
# Step 1: Download & Extract Dataset
# ---------------------------
dataset_url = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
zip_filename = "cats_and_dogs_filtered.zip"
extract_dir = "cats_and_dogs_filtered"

# Download dataset if not exists
if not os.path.exists(zip_filename):
    print("📥 Downloading dataset...")
    urllib.request.urlretrieve(dataset_url, zip_filename)
    print("✅ Download complete.")

# Extract dataset if not already extracted
if not os.path.exists(extract_dir):
    print("📂 Extracting dataset...")
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall(".")
    print("✅ Extraction complete.")

# Define dataset paths
train_cat_folder = os.path.join(extract_dir, "train/cats")
train_dog_folder = os.path.join(extract_dir, "train/dogs")
val_cat_folder = os.path.join(extract_dir, "validation/cats")
val_dog_folder = os.path.join(extract_dir, "validation/dogs")

# Ensure dataset exists
for folder in [train_cat_folder, train_dog_folder, val_cat_folder, val_dog_folder]:
    if not os.path.exists(folder):
        print(f"⚠️ Error: Missing dataset folder: {folder}")
        exit(1)

# ---------------------------
# Step 2: Load Dataset
# ---------------------------
def load_images_from_folder(folder, label, size=(64, 64), limit=500):
    images, labels = [], []
    file_list = os.listdir(folder)[:limit]  # Limit number of images
    
    for filename in file_list:
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            continue  # Skip unreadable images
        img = cv2.resize(img, size)
        images.append(img)
        labels.append(label)
    
    return images, labels

# Load dataset
train_cat_images, train_cat_labels = load_images_from_folder(train_cat_folder, label=0)
val_cat_images, val_cat_labels = load_images_from_folder(val_cat_folder, label=0)
train_dog_images, train_dog_labels = load_images_from_folder(train_dog_folder, label=1)
val_dog_images, val_dog_labels = load_images_from_folder(val_dog_folder, label=1)

# Convert to NumPy arrays
X = np.array(train_cat_images + val_cat_images + train_dog_images + val_dog_images, dtype=np.uint8)
y = np.array(train_cat_labels + val_cat_labels + train_dog_labels + val_dog_labels, dtype=np.int32)

# ---------------------------
# Step 3: Feature Extraction using HOG
# ---------------------------
def extract_features(images):
    feature_list = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        features = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), visualize=False)  # Extract HOG features
        feature_list.append(features)
    return np.array(feature_list, dtype=np.float32)

print("📊 Extracting features...")
X_features = extract_features(X)

# ---------------------------
# Step 4: Train a Model (SVM)
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)
print("🛠 Training SVM model...")
model = SVC(kernel='linear')  
model.fit(X_train, y_train)

# ---------------------------
# Step 5: Evaluate Model
# ---------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy:.2f}")

# ---------------------------
# Step 6: Function to Predict New Image
# ---------------------------
def predict_image(img_path, model):
    if not os.path.exists(img_path):
        print(f"⚠️ Error: File {img_path} not found!")
        return None

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"⚠️ Error: Unable to read image {img_path}")
        return None

    img = cv2.resize(img, (64, 64))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), visualize=False)
    prediction = model.predict([features])[0]
    return "🐱 Cat" if prediction == 0 else "🐶 Dog"

# ---------------------------
# Step 7: Test Image Prediction
# ---------------------------
test_image_path = input("Enter the path of the test image: ").strip()
if os.path.exists(test_image_path):
    prediction = predict_image(test_image_path, model)
    if prediction:
        print(f"🔍 Prediction: {prediction}")

        img = cv2.imread(test_image_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img)
            plt.axis("off")
            plt.title(prediction)
            plt.show()
else:
    print("⚠️ No valid test image found! Please provide a test image.")