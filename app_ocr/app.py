import os
import cv2
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory
import pytesseract
import traceback

# ================== APP SETUP ==================
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ================== TESSERACT CONFIG ==================
# Set path if not in PATH (Windows)
pytesseract.pytesseract.tesseract_cmd = r"D:\Program Files\Tesseract-OCR\tesseract.exe"

# ================== NUMBER ZONE DETECTION ==================
def detect_number_zone_inside_object(object_roi):
    """Detect horizontal number zones inside object."""
    gray = cv2.cvtColor(object_roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Merge digits horizontally
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    # Find horizontal contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    zones = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / h
        if w > 20 and h > 15 and aspect_ratio > 1.0:
            zones.append((x, y, x + w, y + h))

    zones.sort(key=lambda z: z[0])
    return zones

# ================== PREPROCESSING ==================
def preprocess_for_ocr(digit_img):
    gray = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    resized = cv2.resize(clean, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    inverted = cv2.bitwise_not(resized)
    return inverted

# ================== SEGMENT DIGITS BY PROJECTION ==================
def segment_digits(number_zone):
    """Split a number zone into individual digit images using vertical projection."""
    gray = cv2.cvtColor(number_zone, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Sum of pixels in each column
    col_sum = np.sum(thresh, axis=0)
    thresh_col = col_sum > 0

    # Find transitions from 0 → 1 (start of digit) and 1 → 0 (end of digit)
    digits = []
    start = None
    for i, val in enumerate(thresh_col):
        if val and start is None:
            start = i
        elif not val and start is not None:
            end = i
            if end - start > 3:  # ignore tiny gaps
                digit_img = number_zone[:, start:end]
                digits.append(digit_img)
            start = None
    if start is not None:  # last digit
        digit_img = number_zone[:, start:]
        digits.append(digit_img)
    return digits

# ================== IMAGE PROCESSING ==================
def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return "Read Error", 0, None, []

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    h_img, w_img = image.shape[:2]
    output_image = image.copy()

    # --- 1️⃣ Detect object ROI ---
    obj_x1, obj_y1, obj_x2, obj_y2 = int(w_img*0.05), int(h_img*0.05), int(w_img*0.95), int(h_img*0.95)
    object_roi = image[obj_y1:obj_y2, obj_x1:obj_x2]

    # --- 2️⃣ Detect number zones ---
    number_zones = detect_number_zone_inside_object(object_roi)
    if not number_zones:
        n_x1, n_y1, n_x2, n_y2 = 0, int(object_roi.shape[0]*0.4), object_roi.shape[1], int(object_roi.shape[0]*0.6)
        number_zones = [(n_x1, n_y1, n_x2, n_y2)]

    detected_reading = ""
    crop_files = []

    # --- 3️⃣ OCR each zone ---
    for idx, (x1, y1, x2, y2) in enumerate(number_zones):
        zone = object_roi[y1:y2, x1:x2]

        # Segment digits using vertical projection
        digit_imgs = segment_digits(zone)

        zone_output = zone.copy()
        for digit_img in digit_imgs:
            preprocessed = preprocess_for_ocr(digit_img)
            text = pytesseract.image_to_string(
                preprocessed,
                config="--psm 10 -c tessedit_char_whitelist=0123456789"
            ).strip()
            if text == "":
                text = "0"  # fallback if OCR fails for very thin zeros
            detected_reading += text

        # Save cropped zone
        crop_fn = f"crop_{timestamp}_{idx}.jpg"
        cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], crop_fn), zone_output)
        crop_files.append(crop_fn)

        # Draw rectangle on object ROI
        cv2.rectangle(object_roi, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw object ROI on full image
    cv2.rectangle(output_image, (obj_x1, obj_y1), (obj_x2, obj_y2), (255, 0, 0), 2)

    # Save final annotated image
    res_fn = f"result_{timestamp}.jpg"
    cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], res_fn), output_image)

    return detected_reading or "Not Found", 1.0, res_fn, crop_files

# ================== FLASK ROUTES ==================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/upload', methods=['POST'])
def upload():
    try:
        file = request.files['image']
        path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(path)
        reading, conf, res, crops = process_image(path)
        return jsonify({
            'success': True,
            'reading': reading,
            'confidence': f"{conf*100:.1f}%",
            'image_url': f"/uploads/{res}",
            'crops': crops
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

# ================== MAIN ==================
if __name__ == '__main__':
    app.run(debug=True, port=5000)
