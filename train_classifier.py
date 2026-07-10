import sys
sys.modules['bottleneck'] = None
sys.modules['numexpr'] = None

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

print("=====================================================")
print("  DermAlert V2: Advanced GLCM & LBP Texture Engine   ")
print("=====================================================")

DATASET_DIR = "dataset"

def extract_advanced_features(image_path):
    if not image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
        return None
        
    img = cv2.imread(image_path)
    if img is None:
        return None
    try:
        img = cv2.resize(img, (256, 256)) # Standardized size for texture scanning
        
        # 1. Preprocessing Pipeline (CLAHE & Dull-Razor)
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(5, 5))
        cl_y = clahe.apply(y)
        enhanced = cv2.cvtColor(cv2.merge((cl_y, cr, cb)), cv2.COLOR_YCrCb2BGR)
        
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        smooth = cv2.bilateralFilter(gray, 7, 50, 50)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        blackhat = cv2.morphologyEx(smooth, cv2.MORPH_BLACKHAT, kernel)
        _, hair_mask = cv2.threshold(blackhat, 11, 255, cv2.THRESH_BINARY)
        shaved = cv2.inpaint(enhanced, hair_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        shaved_gray = cv2.cvtColor(shaved, cv2.COLOR_BGR2GRAY)
        
        # FEATURE 1: Advanced Color Distributions (6 metrics)
        b, g, r = cv2.split(shaved)
        mean_r, std_r = cv2.meanStdDev(r)
        mean_g, std_g = cv2.meanStdDev(g)
        mean_b, std_b = cv2.meanStdDev(b)
        features = [mean_r[0][0], std_r[0][0], mean_g[0][0], std_g[0][0], mean_b[0][0], std_b[0][0]]
        
        # FEATURE 2: GLCM Texture Descriptors (4 metrics)
        # Calculates spatial relationships between pixel intensities
        glcm = graycomatrix(shaved_gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast')[0][0]
        correlation = graycoprops(glcm, 'correlation')[0][0]
        energy = graycoprops(glcm, 'energy')[0][0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0][0]
        features.extend([contrast, correlation, energy, homogeneity])
        
        # FEATURE 3: Local Binary Patterns (LBP) Histogram (10 metrics)
        # Extracts fine-grained micro-surface structures
        lbp = local_binary_pattern(shaved_gray, P=8, R=1, method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10), density=True)
        features.extend(lbp_hist.tolist())
        
        # FEATURE 4: Shape Geometry (6 metrics)
        blurred = cv2.GaussianBlur(shaved_gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        contour_count = len(contours)
        max_area = max([cv2.contourArea(c) for c in contours]) if contours else 0
        _, std_gray = cv2.meanStdDev(shaved_gray)
        variance_gray = std_gray[0][0] ** 2
        
        features.extend([contour_count, max_area, variance_gray, np.mean(shaved_gray), np.std(shaved_gray), np.min(shaved_gray)])
        
        return features
    except Exception as e:
        return None

X, y = [], []
categories = ["benign", "malignant_inflamed"]

print("\nExtracting GLCM, LBP, and Color vectors from medical dataset...")
for category_idx, category_name in enumerate(categories):
    folder_path = os.path.join(DATASET_DIR, category_name)
    if os.path.exists(folder_path):
        all_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        images = all_files[:300] # Kept at 300 per category for speed optimization
        
        print(f"\n-> Analyzing folder '{category_name}' ({len(images)} files)...")
        for idx, img_name in enumerate(images):
            features = extract_advanced_features(os.path.join(folder_path, img_name))
            if features is not None:
                X.append(features)
                y.append(category_idx)
            
            if (idx + 1) % 50 == 0 or (idx + 1) == len(images):
                print(f"   [Texture Mapping] Processed {idx + 1} / {len(images)} matrices...", flush=True)

if len(X) >= 50:
    X, y = np.array(X), np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    
    print("\nTraining Non-Linear RBF Support Vector Machine Hyperplane...")
    # Using balanced class weights and soft margin scaling for optimized tuning
    svm_model = SVC(kernel='rbf', C=50.0, gamma='scale', probability=True, class_weight='balanced')
    svm_model.fit(X_train, y_train)
    
    print(f"\nOptimization Complete! True Model Accuracy: {accuracy_score(y_test, svm_model.predict(X_test))*100:.2f}%")
    with open("model.pkl", "wb") as f:
        pickle.dump(svm_model, f)
    print("Success: Advanced portfolio 'model.pkl' saved to disk.")
else:
    print("Error: Matrix array compilation failed.")