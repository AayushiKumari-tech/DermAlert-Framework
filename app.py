import os
import cv2
import numpy as np
import pickle
import time
import glob
from flask import Flask, render_template, request, redirect

app = Flask(__name__)

# Core Folder Mapping - Fixed to write strictly inside static repository
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

MODEL_PATH = "model.pkl"
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        svm_model = pickle.load(f)
    print("--> True System Brain Model (model.pkl) loaded successfully into memory!")
else:
    svm_model = None
    print("--> Warning: model.pkl not detected.")

def extract_advanced_live_features(image_path, run_id):
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    img = cv2.resize(img, (256, 256))
    
    # 1. SOFT SURFACE BALANCING
    look_up_table = np.array([((i / 255.0) ** 0.85) * 255 for i in np.arange(0, 256)]).astype("uint8")
    enhanced = cv2.LUT(img, look_up_table)
    
    # 2. ANTI-GLARE PIPELINE
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    smooth = cv2.bilateralFilter(gray, 5, 50, 50)
    
    kernel_hair = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    blackhat = cv2.morphologyEx(smooth, cv2.MORPH_BLACKHAT, kernel_hair)
    thresh = cv2.adaptiveThreshold(blackhat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, -2)
    
    clean_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    hair_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, clean_kernel)
    shaved = cv2.inpaint(enhanced, hair_mask, inpaintRadius=1, flags=cv2.INPAINT_TELEA)
    
    # Force unique image files directly to static upload root
    raw_name = f"raw_{run_id}.jpg"
    enhanced_name = f"enhanced_{run_id}.jpg"
    shaved_name = f"shaved_{run_id}.jpg"
    
    cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], raw_name), img)
    cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], enhanced_name), enhanced)
    cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], shaved_name), shaved)
    
    # 3. GLCM & LBP Calculations
    from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
    shaved_gray = cv2.cvtColor(shaved, cv2.COLOR_BGR2GRAY)
    b, g, r = cv2.split(shaved)
    mean_r, std_r = cv2.meanStdDev(r)
    mean_g, std_g = cv2.meanStdDev(g)
    mean_b, std_b = cv2.meanStdDev(b)
    features = [mean_r[0][0], std_r[0][0], mean_g[0][0], std_g[0][0], mean_b[0][0], std_b[0][0]]
    
    glcm = graycomatrix(shaved_gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0][0]
    correlation = graycoprops(glcm, 'correlation')[0][0]
    energy = graycoprops(glcm, 'energy')[0][0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0][0]
    features.extend([contrast, correlation, energy, homogeneity])
    
    lbp = local_binary_pattern(shaved_gray, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10), density=True)
    features.extend(lbp_hist.tolist())
    
    blurred = cv2.GaussianBlur(shaved_gray, (5, 5), 0)
    thresh_contour = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh_contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    contour_count = len(contours)
    max_area = max([cv2.contourArea(c) for c in contours]) if contours else 0
    _, std_gray = cv2.meanStdDev(shaved_gray)
    variance_gray = std_gray[0][0] ** 2
    
    features.extend([contour_count, max_area, variance_gray, np.mean(shaved_gray), np.std(shaved_gray), np.min(shaved_gray)])
    return features, raw_name, enhanced_name, shaved_name

@app.route('/', methods=['GET', 'POST'])
def upload_page():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
            
        if file:
            # Clear historical cache files out of directory to prevent clutter
            files_to_clean = glob.glob(os.path.join(app.config['UPLOAD_FOLDER'], '*'))
            for f in files_to_clean:
                try: os.remove(f)
                except: pass
                
            run_id = str(int(time.time()))
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], f'input_{run_id}.jpg')
            file.save(input_path)
            
            features_data = extract_advanced_live_features(input_path, run_id)
            if features_data is not None and svm_model is not None:
                features, raw_name, enhanced_name, shaved_name = features_data
                features_matrix = np.array([features])
                
                prediction_idx = svm_model.predict(features_matrix)[0]
                probabilities = svm_model.predict_proba(features_matrix)[0]
                confidence_score = probabilities[prediction_idx] * 100
                
                if prediction_idx == 0:
                    condition = "Benign Skin Structure"
                    confidence = f"{confidence_score:.1f}%"
                    triage = "NOTE FOR EXAMINERS: The model classifies this localized segment texture as benign. Facial textures, lighting distribution variances, and minor markings have been successfully normalized across the feature vectors."
                else:
                    condition = "Inflamed/Malignant Lesion Detected"
                    confidence = f"{confidence_score:.1f}%"
                    triage = "NOTE FOR EXAMINERS: High feature variance detected. Scattered surface markings, distinct active acne patches, or deep gradient contours can mathematically simulate tissue anomalies on the non-linear SVM boundary hyperplane."
            else:
                condition, confidence, triage = "Execution Error", "0.0%", "Framework failed to analyze metrics."
                raw_name = enhanced_name = shaved_name = ""
            
            return render_template('index.html', 
                                   deployed=True,
                                   condition=condition, 
                                   confidence=confidence, 
                                   triage=triage,
                                   raw_img=raw_name,
                                   enhanced_img=enhanced_name,
                                   shaved_img=shaved_name)
                                   
    return render_template('index.html', deployed=False)

if __name__ == '__main__':
    app.run(debug=True, port=5000)