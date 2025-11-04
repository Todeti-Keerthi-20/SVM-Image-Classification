# STEP 1: IMPORT LIBRARIES
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# STEP 2: LOAD IMAGES AND LABELS
data_dir = "dataset"   # Folder containing 'cats' and 'dogs'
categories = ["cats", "dogs"]
img_size = 100

data = []
labels = []

print("📥 Loading images...")

for category in categories:
    path = os.path.join(data_dir, category)

    # Check if folder exists
    if not os.path.exists(path):
        print(f"⚠ Folder not found for category '{category}', skipping...")
        continue

    label = categories.index(category)
    count = 0

    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)

        # Check if file is an image
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        # Read and process image safely
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"❌ Unable to read {img_name}, skipping...")
            continue

        img = cv2.resize(img, (img_size, img_size))
        data.append(img.flatten())
        labels.append(label)
        count += 1

    print(f"✅ Loaded {count} valid images for class: {category}")

# ✅ Added here (end of Step 2)
if len(data) == 0:
    print("❌ No images found! Check dataset path and structure.")
    exit()

print("✅ Image loading complete!")

# STEP 3: CONVERT TO NUMPY ARRAYS
X = np.array(data)
y = np.array(labels)

print("Total features (X) shape:", X.shape)
print("Total labels (y) shape:", y.shape)

# STEP 4: SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("✅ Data successfully split into training and testing sets.")

# STEP 5: TRAIN SVM MODEL
print("⚙ Training SVM model...")
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)
print("✅ Model training complete!")

# STEP 6: TEST & ACCURACY
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"🎯 Model Accuracy: {acc*100:.2f}%")

# STEP 7: TEST WITH NEW IMAGE (optional)
test_image = os.path.join(data_dir, "cats", os.listdir(os.path.join(data_dir, "cats"))[0])
if os.path.exists(test_image):
    img = cv2.imread(test_image, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (100, 100))
    img_flat = img.flatten().reshape(1, -1)
    pred = model.predict(img_flat)
    print("🐾 Prediction for test image:", categories[pred[0]])
else:
    print("Test image not found, skipping prediction.")