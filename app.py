from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import numpy as np
import os
from datetime import datetime
from keras.models import load_model
import traceback

app = Flask(__name__)

# Upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load CNN model for digit recognition
try:
    cnn_model = load_model("model.keras", compile=False)
    print("✅ CNN model loaded successfully")
except Exception as e:
    cnn_model = None
    print(f"⚠️ CNN model not loaded: {e}")

def preprocess_digit(digit_img):
    """Prepare digit for CNN"""
    gray = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    resized = cv2.resize(gray, (28, 28))
    return resized.astype('float32') / 255.0

def extract_digits_from_roi(roi):
    """
    Extract individual digits from a fixed ROI using contours
    Returns list of digit crops sorted left-to-right
    """
    # Threshold and clean
    _, thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digits = []

    h_roi, w_roi = roi.shape
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h > h_roi * 0.4 and w > w_roi * 0.05:  # filter noise
            digits.append((x, y, w, h))

    # Sort left to right
    digits = sorted(digits, key=lambda b: b[0])
    digit_crops = [roi[y:y+h, x:x+w] for x, y, w, h in digits]

    return digits, digit_crops

def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None: return "Error", 0, None, []
    
    h_img, w_img = image.shape[:2]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output = image.copy()

    # 1. BROADEN THE SEARCH AREA (Ensure we don't miss the numbers)
    mask = np.zeros(image.shape[:2], dtype="uint8")
    # This covers the middle-upper section of the dial
    cv2.rectangle(mask, (int(w_img*0.20), int(h_img*0.25)), 
                        (int(w_img*0.80), int(h_img*0.55)), 255, -1)

    # 2. IMPROVED THRESHOLDING
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    masked_gray = cv2.bitwise_and(gray, gray, mask=mask)
    
    # Using Otsu's Binarization is often more stable for black text on white
    _, thresh = cv2.threshold(masked_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 3. FIND DIGITS
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digit_boxes = []
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = h / float(w)
        
        # Check for vertical rectangles that look like digits
        # Adjusted height requirement to be more flexible
        if 1.1 < aspect_ratio < 4.5 and h > (h_img * 0.03):
            digit_boxes.append((x, y, w, h))

    # Sort Left-to-Right
    digit_boxes = sorted(digit_boxes, key=lambda b: b[0])

    # 4. DRAW GREEN BOXES AND RECOGNIZE
    detected_reading = ""
    confidences = []

    for (x, y, w, h) in digit_boxes:
        # DRAW THE GREEN BOX
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # CNN Recognition
        if cnn_model:
            digit_crop = image[y:y+h, x:x+w]
            crop_rgb = cv2.cvtColor(digit_crop, cv2.COLOR_BGR2RGB)
            crop_resized = cv2.resize(crop_rgb, (20, 32))
            crop_resized = crop_resized.astype('float32') / 255.0
            crop_resized = np.expand_dims(crop_resized, axis=0)
            
            pred = cnn_model.predict(crop_resized, verbose=0)
            digit = np.argmax(pred)
            detected_reading += str(digit)
            confidences.append(float(np.max(pred)))
            
            # Label number above the box
            cv2.putText(output, str(digit), (x, y - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 5. CREATE CROP FROM THE LABELED OUTPUT
    if digit_boxes:
        min_x = max(0, digit_boxes[0][0] - 10)
        max_x = min(w_img, digit_boxes[-1][0] + digit_boxes[-1][2] + 10)
        min_y = max(0, min([b[1] for b in digit_boxes]) - 10)
        max_y = min(h_img, max([b[1] + b[3] for b in digit_boxes]) + 10)
        crop_zone = output[min_y:max_y, min_x:max_x]
    else:
        # If still not found, show the masked area to debug
        crop_zone = cv2.bitwise_and(output, output, mask=mask)
        detected_reading = "Not Found"

    # SAVE BOTH
    crop_filename = f"crop_{timestamp}.jpg"
    result_filename = f"result_{timestamp}.jpg"
    cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], crop_filename), crop_zone)
    cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], result_filename), output)

    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    return detected_reading, avg_conf, result_filename, [crop_filename]
    
# Routes
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
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        allowed_extensions = {'png', 'jpg', 'jpeg', 'bmp'}
        if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
            return jsonify({'error': 'Invalid file type. Use PNG, JPG, JPEG, BMP.'}), 400

        # Save uploaded image
        original_filename = f"original_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
        file.save(original_path)

        # Process image
        reading, confidence, result_filename, digit_filenames = process_image(original_path)
        if result_filename is None:
            return jsonify({'error': reading}), 500

        image_url = f"/uploads/{result_filename}"
        digit_urls = [f"/uploads/{f}" for f in digit_filenames]

        return jsonify({
            'success': True,
            'reading': reading,
            'confidence': f"{confidence:.2%}",
            'image_url': image_url,
            'digit_urls': digit_urls,
            'filename': result_filename
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Water Meter Recognition App...")
    app.run(debug=True, port=5000)
