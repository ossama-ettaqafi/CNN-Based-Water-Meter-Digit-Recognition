from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import numpy as np
import os
import base64
from datetime import datetime
from keras.models import load_model

app = Flask(__name__)

# Create upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Allow Flask to serve files from uploads folder
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model
try:
    model = load_model("model.keras", compile=False)
    print("✅ Model loaded successfully")
except Exception as e:
    model = None
    print(f"⚠️ Model not loaded: {e}")

def predict_digit(img_digit):
    """Predict a single digit"""
    if model is None:
        # Mock prediction for testing
        return np.random.randint(0, 10), np.random.random()
    
    # Convert to RGB if grayscale
    if len(img_digit.shape) == 2:
        img_digit = cv2.cvtColor(img_digit, cv2.COLOR_GRAY2RGB)
    
    # Resize for model
    img_digit = cv2.resize(img_digit, (20, 32))
    img_digit = img_digit.astype('float32') / 255.0
    img_digit = np.expand_dims(img_digit, axis=0)
    
    # Predict
    probs = model.predict(img_digit, verbose=0)
    return np.argmax(probs), float(np.max(probs))

def process_image(image_path):
    """Process the water meter image"""
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        return "Error: Could not read image", 0.0, None
    
    # Make copy for drawing
    output = image.copy()
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Define ROI (adjust these values for your meter)
    h, w = enhanced.shape
    roi_y, roi_h = int(h * 0.35), int(h * 0.12)
    roi_x, roi_w = int(w * 0.48), int(w * 0.35)
    
    # Extract ROI
    roi = enhanced[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
    
    # Threshold
    _, thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Clean up
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours
    digits = []
    for contour in contours:
        x, y, wc, hc = cv2.boundingRect(contour)
        if hc > roi_h * 0.4 and wc > roi_w * 0.05:
            digits.append((x, y, wc, hc))
    
    # Sort left to right
    digits.sort(key=lambda c: c[0])
    
    # Process each digit
    result = ""
    confidences = []
    
    for x, y, wc, hc in digits[:8]:  # Take max 8 digits
        # Extract digit
        digit_img = roi[y:y+hc, x:x+wc]
        
        # Threshold digit
        _, digit_thresh = cv2.threshold(digit_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Add border
        margin = 5
        digit_with_border = np.zeros((hc + 2*margin, wc + 2*margin), dtype=np.uint8)
        digit_with_border[margin:margin+hc, margin:margin+wc] = digit_thresh
        
        # Ensure white digits on black background
        if np.mean(digit_with_border) < 127:
            digit_with_border = cv2.bitwise_not(digit_with_border)
        
        # Convert to RGB
        digit_rgb = cv2.cvtColor(digit_with_border, cv2.COLOR_GRAY2RGB)
        
        # Predict
        digit, confidence = predict_digit(digit_rgb)
        result += str(digit)
        confidences.append(confidence)
        
        # Draw on output image
        cv2.rectangle(output, 
                     (roi_x + x, roi_y + y), 
                     (roi_x + x + wc, roi_y + y + hc), 
                     (0, 255, 0), 2)
        cv2.putText(output, str(digit), 
                   (roi_x + x, roi_y + y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Draw ROI rectangle
    cv2.rectangle(output, 
                 (roi_x, roi_y), 
                 (roi_x + roi_w, roi_y + roi_h), 
                 (255, 0, 0), 2)
    
    # Generate unique filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_filename = f"result_{timestamp}.jpg"
    result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
    
    # Save result
    cv2.imwrite(result_path, output)
    print(f"✅ Saved processed image to: {result_path}")
    
    # Calculate average confidence
    avg_confidence = np.mean(confidences) if confidences else 0.0
    
    return result, avg_confidence, result_filename

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/upload', methods=['POST'])
def upload():
    """Handle image upload"""
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Check file extension
    allowed_extensions = {'png', 'jpg', 'jpeg', 'bmp'}
    if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
        return jsonify({'error': 'Invalid file type. Use PNG, JPG, or JPEG.'}), 400
    
    # Save original file
    original_filename = f"original_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
    original_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
    file.save(original_path)
    print(f"✅ Saved original image to: {original_path}")
    
    # Process image
    reading, confidence, result_filename = process_image(original_path)
    
    # Build full URL for the image
    image_url = f"/uploads/{result_filename}"
    
    return jsonify({
        'success': True,
        'reading': reading,
        'confidence': f"{confidence:.2%}",
        'image_url': image_url,
        'filename': result_filename
    })

@app.route('/cleanup', methods=['POST'])
def cleanup():
    """Clean up uploaded files (optional)"""
    try:
        import glob
        import time
        
        # Delete files older than 1 hour
        files = glob.glob(os.path.join(UPLOAD_FOLDER, '*'))
        current_time = time.time()
        deleted = 0
        
        for file in files:
            if os.path.getctime(file) < current_time - 3600:  # 1 hour
                os.remove(file)
                deleted += 1
        
        return jsonify({'success': True, 'deleted': deleted})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Water Meter Recognition App...")
    print(f"📁 Upload folder: {os.path.abspath(UPLOAD_FOLDER)}")
    print(f"🤖 Model loaded: {model is not None}")
    print("🌐 Open http://localhost:5000 in your browser")
    app.run(debug=True, port=5000)