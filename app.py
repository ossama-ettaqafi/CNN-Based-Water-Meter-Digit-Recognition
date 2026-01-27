from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import numpy as np
import os
from datetime import datetime
import pytesseract
import re
from collections import Counter

# ================== TESSERACT CONFIG ==================
pytesseract.pytesseract.tesseract_cmd = r"D:\Program Files\Tesseract-OCR\tesseract.exe"

# ================== FLASK APP ==================
app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
DEBUG_FOLDER = "debug"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DEBUG_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["DEBUG_FOLDER"] = DEBUG_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max file size

# =====================================================
# SMART DIGIT DETECTION WITH VISUALIZATION
# =====================================================
def detect_digits_with_boxes(image_roi, roi_coords, full_image):
    """Detect individual digits and return their positions and values"""
    x1, y1, x2, y2 = roi_coords
    h_roi, w_roi = image_roi.shape
    
    # Enhance the ROI for better detection
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image_roi)
    
    # Threshold to get dark digits on light background
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Clean up noise
    kernel = np.ones((2, 2), np.uint8)
    binary_clean = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary_clean = cv2.morphologyEx(binary_clean, cv2.MORPH_OPEN, kernel)
    
    # Find contours of potential digits
    contours, _ = cv2.findContours(binary_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    digit_info = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        
        # Filter for digit-like shapes
        if (h > h_roi * 0.3 and  # Minimum height relative to ROI
            w > w_roi * 0.03 and   # Minimum width
            area > 50 and          # Minimum area
            0.2 < w/h < 1.2):     # Reasonable aspect ratio for digits
            
            # Add padding around the digit
            pad_x = int(w * 0.2)
            pad_y = int(h * 0.2)
            
            # Ensure we don't go out of bounds
            x_start = max(0, x - pad_x)
            y_start = max(0, y - pad_y)
            x_end = min(w_roi, x + w + pad_x)
            y_end = min(h_roi, y + h + pad_y)
            
            # Extract digit region
            digit_roi = enhanced[y_start:y_end, x_start:x_end]
            
            # OCR on the individual digit
            config_digit = "--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789"
            digit_text = pytesseract.image_to_string(digit_roi, config=config_digit).strip()
            
            # If OCR fails, try with different preprocessing
            if not digit_text.isdigit():
                # Try with binarization
                _, digit_binary = cv2.threshold(digit_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                digit_text = pytesseract.image_to_string(digit_binary, config=config_digit).strip()
            
            if digit_text and digit_text.isdigit():
                digit = digit_text[0]
                confidence = 85.0  # Base confidence for detected digits
            else:
                digit = "?"
                confidence = 30.0
            
            # Calculate absolute coordinates on full image
            abs_x = x1 + x_start
            abs_y = y1 + y_start
            abs_w = x_end - x_start
            abs_h = y_end - y_start
            
            digit_info.append({
                'x': abs_x,
                'y': abs_y,
                'w': abs_w,
                'h': abs_h,
                'center_x': abs_x + abs_w // 2,
                'digit': digit,
                'confidence': confidence,
                'roi': digit_roi
            })
    
    # Sort digits by their horizontal position (left to right)
    digit_info.sort(key=lambda d: d['center_x'])
    
    return digit_info

def draw_digit_boxes(full_image, digit_info, reading):
    """Draw green boxes around detected digits and label them"""
    output = full_image.copy()
    
    # Draw green boxes for each detected digit
    for i, digit_data in enumerate(digit_info):
        x, y, w, h = digit_data['x'], digit_data['y'], digit_data['w'], digit_data['h']
        
        # Draw green rectangle around digit
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Label with digit value
        label = f"{digit_data['digit']}"
        cv2.putText(output, label,
                   (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, (0, 255, 0), 2)
        
        # Optional: Add index number
        cv2.putText(output, f"#{i+1}",
                   (x + w - 20, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, (0, 200, 0), 1)
    
    # Also draw the complete reading at the top
    if reading:
        cv2.putText(output, f"Reading: {reading}",
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                   1.2, (0, 255, 255), 3)
    
    return output

# =====================================================
# SMART READING EXTRACTION WITH DIGIT DETECTION
# =====================================================
def extract_meter_reading_with_digits(image):
    """Smart extraction with individual digit detection and visualization"""
    h, w = image.shape[:2]
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Define the ROI where digits are expected (based on your meter layout)
    # Adjust these values based on your specific meter images
    roi_y1, roi_y2 = int(h * 0.22), int(h * 0.38)
    roi_x1, roi_x2 = int(w * 0.25), int(w * 0.75)
    
    roi = gray[roi_y1:roi_y2, roi_x1:roi_x2]
    roi_coords = (roi_x1, roi_y1, roi_x2, roi_y2)
    
    # Save debug ROI
    debug_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    debug_path = os.path.join(DEBUG_FOLDER, f"{debug_timestamp}_roi.jpg")
    cv2.imwrite(debug_path, roi)
    
    # Detect individual digits with their positions
    digit_info = detect_digits_with_boxes(roi, roi_coords, image)
    
    # Extract the reading from detected digits
    detected_digits = ''.join([d['digit'] for d in digit_info if d['digit'].isdigit()])
    
    # Try OCR on the whole ROI as fallback
    config_full = "--psm 6 --oem 3"
    full_text = pytesseract.image_to_string(roi, config=config_full)
    
    # Look for patterns in the full text
    reading = ""
    confidence = 0.0
    
    # Pattern 1: Look for digits followed by m³
    pattern_m3 = re.search(r'(\d[\d\s]*?)\s*m³', full_text)
    if pattern_m3:
        digits = re.sub(r'\s', '', pattern_m3.group(1))
        reading = digits.rjust(8, "0")[:8]
        confidence = 80.0
        print(f"Found with m³ pattern: {reading}")
    
    # Pattern 2: Look for 8 consecutive digits
    elif re.search(r'(\d{8})', full_text):
        match = re.search(r'(\d{8})', full_text)
        reading = match.group(1)
        confidence = 75.0
        print(f"Found 8-digit pattern: {reading}")
    
    # Pattern 3: Use detected individual digits
    elif len(detected_digits) >= 6:
        reading = detected_digits[:8].ljust(8, "0")
        confidence = min(90.0, 70.0 + len(detected_digits) * 3)
        print(f"Using detected digits: {reading}")
    
    # Pattern 4: Digits-only OCR
    else:
        config_digits = "--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789 "
        digits_text = pytesseract.image_to_string(roi, config=config_digits).replace(" ", "")
        if len(digits_text) >= 6:
            reading = digits_text[:8].ljust(8, "0")
            confidence = 65.0
            print(f"Digits-only OCR: {reading}")
        else:
            reading = "00000000"
            confidence = 30.0
            print(f"No reliable reading found, using default: {reading}")
    
    # Create visualization with green boxes
    output_image = draw_digit_boxes(image, digit_info, reading)
    
    # Draw the main ROI rectangle
    cv2.rectangle(output_image, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 2)
    cv2.putText(output_image, "Reading Area", 
               (roi_x1, roi_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
               0.7, (255, 0, 0), 2)
    
    return reading, confidence, output_image, digit_info, debug_timestamp

# =====================================================
# MAIN PROCESSING
# =====================================================
def process_image(image_path):
    """Main processing function"""
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not read image")
    
    h, w = image.shape[:2]
    
    # Extract reading with digit detection
    reading, confidence, output_image, digit_info, debug_timestamp = extract_meter_reading_with_digits(image)
    
    print(f"Final reading: {reading}, Confidence: {confidence}")
    print(f"Detected {len(digit_info)} digit(s)")
    
    # Add confidence text
    cv2.putText(output_image, f"Confidence: {confidence:.1f}%",
               (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
               0.8, (0, 255, 0), 2)
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(output_image, f"Processed: {timestamp}",
               (20, h-20), cv2.FONT_HERSHEY_SIMPLEX,
               0.5, (200, 200, 200), 1)
    
    # Add counter for detected digits
    cv2.putText(output_image, f"Digits detected: {len(digit_info)}",
               (w - 200, 30), cv2.FONT_HERSHEY_SIMPLEX,
               0.6, (0, 200, 255), 2)
    
    # Save result
    filename = f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    result_path = os.path.join(UPLOAD_FOLDER, filename)
    cv2.imwrite(result_path, output_image)
    
    # Prepare digit details for response
    digit_details = []
    for i, digit in enumerate(reading):
        # Find if this digit was individually detected
        detected = False
        digit_confidence = confidence * 0.9  # Start with base confidence
        
        # Check if we have a detected digit at this position
        if i < len(digit_info) and digit_info[i]['digit'].isdigit():
            if digit_info[i]['digit'] == digit:
                detected = True
                digit_confidence = digit_info[i]['confidence']
        
        digit_details.append({
            'position': i+1,
            'digit': digit,
            'confidence': f"{digit_confidence:.1f}%",
            'detected': detected,
            'status': '✓' if detected else '⚠' if digit_confidence > 60 else '✗',
            'box_coordinates': {
                'x': digit_info[i]['x'] if i < len(digit_info) else 0,
                'y': digit_info[i]['y'] if i < len(digit_info) else 0,
                'width': digit_info[i]['w'] if i < len(digit_info) else 0,
                'height': digit_info[i]['h'] if i < len(digit_info) else 0
            } if i < len(digit_info) else None
        })
    
    return {
        'reading': reading,
        'confidence': f"{confidence:.1f}%",
        'image_url': f"/uploads/{filename}",
        'debug_url': f"/debug/{debug_timestamp}_roi.jpg",
        'digit_details': digit_details,
        'digit_count': len(digit_info),
        'timestamp': timestamp,
        'debug_timestamp': debug_timestamp
    }

# =====================================================
# FLASK ROUTES (remain the same)
# =====================================================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/uploads/<filename>")
def uploaded(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route("/debug/<filename>")
def debug_file(filename):
    return send_from_directory(DEBUG_FOLDER, filename)

@app.route("/upload", methods=["POST"])
def upload():
    try:
        file = request.files.get("image")
        if not file:
            return jsonify({"error": "No image provided"}), 400
        
        # Validate file type
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            return jsonify({"error": "Invalid file type. Use PNG, JPG, or BMP"}), 400
        
        # Save uploaded file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        input_name = f"input_{timestamp}.jpg"
        input_path = os.path.join(UPLOAD_FOLDER, input_name)
        file.save(input_path)
        
        # Process image
        result_data = process_image(input_path)
        
        return jsonify({
            "success": True,
            **result_data
        })
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error: {e}\n{error_trace}")
        return jsonify({"error": f"Processing error: {str(e)}"}), 500

@app.route("/health")
def health():
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

@app.route("/debug_images")
def list_debug_images():
    debug_files = []
    for file in os.listdir(DEBUG_FOLDER):
        if file.endswith('.jpg'):
            debug_files.append({
                'name': file,
                'url': f'/debug/{file}',
                'size': os.path.getsize(os.path.join(DEBUG_FOLDER, file))
            })
    
    return jsonify({
        'debug_files': sorted(debug_files, key=lambda x: x['name'], reverse=True)[:10]
    })

# =====================================================
if __name__ == "__main__":
    print("🚰 SMART WATER METER OCR WITH DIGIT DETECTION")
    print("🌐 http://127.0.0.1:5000")
    print("📁 Upload folder:", os.path.abspath(UPLOAD_FOLDER))
    print("💡 Features:")
    print("   - Individual digit detection with green boxes")
    print("   - 8-digit meter reading extraction")
    print("   - Confidence scoring per digit")
    print("   - Visual feedback for detection")
    
    app.run(debug=True, host='0.0.0.0', port=5000)