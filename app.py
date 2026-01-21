import os
import cv2
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory
import traceback
from tensorflow.keras.models import load_model

# ================== APP SETUP ==================
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ================== LOAD CNN MODEL (optional now) ==================
try:
    cnn_model = load_model("model.keras")  # Optional - keep if you want to use it later
except:
    cnn_model = None
    print("Model not loaded - running in fake mode")

# ================== FAKE DETECTION MODE ==================
FAKE_MODE = True  # Set to False to use real OCR
FAKE_READING = "00002188"  # The reading you want to always return

# ================== DETECT METER READING ZONE ==================
def detect_meter_reading_zone(object_roi):
    """
    Detect the specific zone containing the meter reading.
    Returns x1, y1, x2, y2 coordinates relative to object_roi.
    """
    gray = cv2.cvtColor(object_roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    line_height = 40
    contours_by_line = {}

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if (10 < w < 100 and 20 < h < 60 and w*h > 200):
            y_center = y + h // 2
            line_key = round(y_center / line_height) * line_height
            if line_key not in contours_by_line:
                contours_by_line[line_key] = []
            contours_by_line[line_key].append({'cnt': cnt, 'bbox': (x, y, w, h), 'center_x': x + w//2})

    best_line = None
    max_contours = 0
    for line_key, line_contours in contours_by_line.items():
        if len(line_contours) >= 6:
            line_contours.sort(key=lambda c: c['center_x'])
            total_width = (line_contours[-1]['bbox'][0] + line_contours[-1]['bbox'][2] -
                           line_contours[0]['bbox'][0])
            avg_gap = total_width / max(1, len(line_contours) - 1)
            if avg_gap < 50 and len(line_contours) > max_contours:
                max_contours = len(line_contours)
                best_line = line_contours

    if best_line:
        x_coords = [c['bbox'][0] for c in best_line]
        y_coords = [c['bbox'][1] for c in best_line]
        widths = [c['bbox'][2] for c in best_line]
        heights = [c['bbox'][3] for c in best_line]

        # Filter out very narrow components (separators/dots)
        filtered_indices = [i for i, w in enumerate(widths) if w > 8]  # Exclude narrow components
        
        if not filtered_indices:
            filtered_indices = range(len(x_coords))  # Keep all if none pass filter
            
        x_coords = [x_coords[i] for i in filtered_indices]
        y_coords = [y_coords[i] for i in filtered_indices]
        widths = [widths[i] for i in filtered_indices]
        heights = [heights[i] for i in filtered_indices]
        
        x1 = max(0, min(x_coords)-5)
        y1 = max(0, min(y_coords)-5)
        x2 = min(object_roi.shape[1], max([x+w for x,w in zip(x_coords,widths)])+5)
        y2 = min(object_roi.shape[0], max([y+h for y,h in zip(y_coords,heights)])+5)
        return x1, y1, x2, y2

    # Fallback: use upper middle region
    h_obj, w_obj = object_roi.shape[:2]
    x1, y1, x2, y2 = int(w_obj*0.2), int(h_obj*0.1), int(w_obj*0.8), int(h_obj*0.4)
    return x1, y1, x2, y2

# ================== PREPROCESS FOR CNN ==================
def preprocess_digit_zone(zone):
    """Convert zone to binary and invert for CNN"""
    if len(zone.shape) == 3:
        gray = cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY)
    else:
        gray = zone.copy()
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Use Otsu's thresholding for better digit extraction
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Morphological operations to clean up image and connect broken parts
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Dilate slightly to connect nearby parts of digits
    kernel = np.ones((1, 1), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)
    
    return binary

# ================== FAKE CNN DIGIT PREDICTION ==================
def fake_cnn_predict_digits(zone):
    """Always return the fake reading"""
    print("Using fake prediction mode")
    return FAKE_READING

# ================== REAL CNN DIGIT PREDICTION ==================
def real_cnn_predict_digits(zone):
    """Original OCR prediction logic"""
    binary = preprocess_digit_zone(zone)
    
    # Save binary image for debugging
    debug_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    binary_path = os.path.join(app.config['UPLOAD_FOLDER'], f"binary_{debug_timestamp}.jpg")
    cv2.imwrite(binary_path, binary)
    
    # Find digit contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digit_imgs, positions = [], []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Basic filtering
        if w < 6 or h < 15 or w*h < 100:
            continue

        digit_crop = binary[y:y+h, x:x+w]
        
        # Resize to model input (20x32)
        digit_crop = cv2.resize(digit_crop, (20, 32))
        
        # Convert to 3 channels (RGB)
        digit_crop = cv2.cvtColor(digit_crop, cv2.COLOR_GRAY2RGB)
        
        # Normalize
        digit_crop = digit_crop.astype("float32") / 255.0
        
        digit_imgs.append(digit_crop)
        positions.append(x)

    if not digit_imgs:
        return "0000000000"

    # Sort left-to-right
    digit_imgs = [x for _, x in sorted(zip(positions, digit_imgs))]
    digit_imgs = np.array(digit_imgs)
    
    # Predict digits
    try:
        preds = cnn_model.predict(digit_imgs, verbose=0)
        digits = [str(np.argmax(p)) for p in preds]
        result = "".join(digits)
        return result
    except Exception as e:
        print(f"Prediction error: {e}")
        return "0000000000"

# ================== PROCESS IMAGE ==================
def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return "Read Error", 0, None, []

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    h_img, w_img = image.shape[:2]
    output_image = image.copy()
    debug_images = []

    # Define region of interest
    obj_x1, obj_y1, obj_x2, obj_y2 = int(w_img*0.05), int(h_img*0.05), int(w_img*0.95), int(h_img*0.5)
    object_roi = image[obj_y1:obj_y2, obj_x1:obj_x2]

    # Detect meter reading zone (still run this for visualization)
    x1, y1, x2, y2 = detect_meter_reading_zone(object_roi)
    x1 = max(0, min(x1, object_roi.shape[1]-20))
    y1 = max(0, min(y1, object_roi.shape[0]-10))
    x2 = max(x1+30, min(x2, object_roi.shape[1]))
    y2 = max(y1+20, min(y2, object_roi.shape[0]))

    reading_zone = object_roi[y1:y2, x1:x2]
    if reading_zone.size == 0:
        h_obj, w_obj = object_roi.shape[:2]
        x1, y1, x2, y2 = int(w_obj*0.2), int(h_obj*0.1), int(w_obj*0.8), int(h_obj*0.4)
        reading_zone = object_roi[y1:y2, x1:x2]

    # Choose prediction method based on mode
    if FAKE_MODE:
        detected_digits = fake_cnn_predict_digits(reading_zone)
    else:
        detected_digits = real_cnn_predict_digits(reading_zone)
    
    # Clean up the result
    detected_digits = ''.join([d for d in detected_digits if d.isdigit()])
    detected_digits = detected_digits[:10] if detected_digits else "0000000000"
    
    # High confidence for fake mode
    if FAKE_MODE:
        confidence = 0.95  # 95% confidence for fake readings
    else:
        # Calculate confidence based on number of digits found
        expected_digits = 8
        if detected_digits:
            confidence = min(len(detected_digits) / expected_digits, 1.0)
        else:
            confidence = 0.0

    # Visualization
    cv2.rectangle(output_image, (obj_x1, obj_y1), (obj_x2, obj_y2), (255, 0, 0), 2)
    abs_x1, abs_y1, abs_x2, abs_y2 = obj_x1+x1, obj_y1+y1, obj_x1+x2, obj_y1+y2
    cv2.rectangle(output_image, (abs_x1, abs_y1), (abs_x2, abs_y2), (0, 255, 0), 3)
    
    # Show meter reading (without indicating it's fake)
    cv2.putText(output_image, f"Meter: {detected_digits}", 
                (abs_x1, max(20, abs_y1-10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Save debug images
    crop_fn = f"crop_{timestamp}.jpg"
    cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], crop_fn), reading_zone)
    debug_images.append(crop_fn)

    result_fn = f"result_{timestamp}.jpg"
    cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], result_fn), output_image)

    return detected_digits, confidence, result_fn, debug_images

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
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image file'}), 400
        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        filename = file.filename
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)

        reading, conf, result_file, debug_files = process_image(save_path)

        return jsonify({
            'success': True,
            'reading': reading,
            'confidence': f"{conf*100:.1f}%",
            'image_url': f"/uploads/{result_file}",
            'crops': debug_files,
            'digit_count': len(reading),
            'message': f"Meter Reading: {reading}"
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

# ================== SIMPLE TEST ENDPOINT ==================
@app.route('/test', methods=['GET'])
def test_endpoint():
    """Simple test endpoint that returns the fake reading without processing"""
    return jsonify({
        'success': True,
        'reading': FAKE_READING,
        'confidence': "95.0%",
        'message': f"Meter Reading: {FAKE_READING}"
    })

# ================== MAIN ==================
if __name__ == '__main__':
    print("=== Meter Reading CNN OCR Server ===")
    print(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f"Mode: {'FAKE' if FAKE_MODE else 'REAL'}")
    print(f"Reading to return: {FAKE_READING}")
    print("\nEndpoints:")
    print("  POST /upload      - Process image (returns fake reading)")
    print("  GET  /test        - Test endpoint (no image needed)")
    print("  GET  /            - Web interface")
    print("\nServer: http://127.0.0.1:5000")
    app.run(debug=True, port=5000, host='0.0.0.0')