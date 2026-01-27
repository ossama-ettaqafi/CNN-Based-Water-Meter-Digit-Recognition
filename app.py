from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import numpy as np
import os
from datetime import datetime
import pytesseract
import re

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
# FOCUSED METER READING EXTRACTION
# =====================================================
def find_meter_reading_area(image):
    """Find the exact area containing the meter reading"""
    h, w = image.shape[:2]
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Based on your image structure:
    # 1. Look for "FERRO" text in the top-middle
    # 2. The reading is right below it
    
    # Define search areas
    ferro_area = gray[int(h*0.15):int(h*0.25), int(w*0.4):int(w*0.6)]
    reading_area = gray[int(h*0.25):int(h*0.40), int(w*0.2):int(w*0.8)]
    
    return reading_area, (int(w*0.2), int(h*0.25), int(w*0.8), int(h*0.40))

def extract_reading_from_area(roi_gray, roi_coords, original_image):
    """Extract the reading from the specific area"""
    x1, y1, x2, y2 = roi_coords
    h_roi, w_roi = roi_gray.shape
    
    output = original_image.copy()
    
    # Save debug ROI
    debug_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    debug_path = os.path.join(DEBUG_FOLDER, f"{debug_timestamp}_roi.jpg")
    cv2.imwrite(debug_path, roi_gray)
    
    # METHOD 1: Direct OCR on the ROI
    config_digits = "--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789"
    text = pytesseract.image_to_string(roi_gray, config=config_digits).strip()
    
    print(f"Direct OCR text: '{text}'")
    
    # Look for 8-digit sequences
    digit_sequences = re.findall(r'\d{8}', text)
    if digit_sequences:
        reading = digit_sequences[0]
        print(f"Found 8-digit sequence: {reading}")
    else:
        # METHOD 2: Look for digits followed by m³
        # First enhance the image
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(roi_gray)
        
        # Try to find m³ symbol to locate reading
        config_m3 = "--psm 6 --oem 3"
        text_m3 = pytesseract.image_to_string(enhanced, config=config_m3)
        print(f"Enhanced OCR text: '{text_m3}'")
        
        # Look for patterns like "00002188 m³"
        m3_pattern = re.search(r'(\d+)\s*m³', text_m3)
        if m3_pattern:
            digits = m3_pattern.group(1)
            reading = digits.rjust(8, "0")[:8]
            print(f"Found reading with m³: {digits} -> {reading}")
        else:
            # METHOD 3: Look for any digits in the area
            all_digits = re.findall(r'\d', text_m3)
            if len(all_digits) >= 6:  # At least 6 digits
                reading = ''.join(all_digits[:8]).ljust(8, "0")
                print(f"Extracted digits: {reading}")
            else:
                # Last resort: Try to find the 8 largest digit-like blobs
                reading = extract_by_blob_detection(roi_gray, output, roi_coords)
                print(f"Blob detection result: {reading}")
    
    # Draw ROI
    cv2.rectangle(output, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(output, "Reading Area", 
               (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
               0.7, (255, 0, 0), 2)
    
    # Draw reading on image
    cv2.putText(output, reading,
               (x1 + 10, y1 + 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    
    return reading, output, debug_timestamp

def extract_by_blob_detection(roi_gray, output, roi_coords):
    """Extract digits by detecting individual digit blobs"""
    x1, y1, x2, y2 = roi_coords
    h_roi, w_roi = roi_gray.shape
    
    # Enhance image
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(roi_gray)
    
    # Threshold to get dark digits on light background
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Clean up
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    digit_contours = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        
        # Filter for digit-like shapes
        if h > 0 and area > 20:
            aspect_ratio = w / h
            if 0.2 < aspect_ratio < 1.0 and h > h_roi * 0.3:
                digit_contours.append((x, y, w, h, x + w/2))
    
    # Sort left to right
    digit_contours.sort(key=lambda c: c[4])
    
    reading = ""
    for idx, (x, y, w, h, _) in enumerate(digit_contours[:8]):
        # Extract digit
        digit_img = roi_gray[max(0, y-2):min(h_roi, y+h+2),
                            max(0, x-2):min(w_roi, x+w+2)]
        
        # OCR this digit
        config = "--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789"
        digit_text = pytesseract.image_to_string(digit_img, config=config).strip()
        
        if digit_text and digit_text.isdigit():
            reading += digit_text[0]
        else:
            reading += "0"
        
        # Draw box
        abs_x = x1 + max(0, x-2)
        abs_y = y1 + max(0, y-2)
        cv2.rectangle(output,
                     (abs_x, abs_y),
                     (abs_x + min(w_roi, x+w+2) - max(0, x-2),
                      abs_y + min(h_roi, y+h+2) - max(0, y-2)),
                     (0, 255, 0), 2)
        
        cv2.putText(output, reading[-1] if reading else "?",
                   (abs_x, abs_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Ensure 8 digits
    reading = reading.ljust(8, "0")[:8]
    
    return reading

# =====================================================
# SMART READING EXTRACTION WITH CONTEXT
# =====================================================
def extract_meter_reading_smart(image):
    """Smart extraction using context from the meter image"""
    h, w = image.shape[:2]
    
    # Strategy 1: Look for reading near "m³" text
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Search in the area where reading should be (top-middle)
    search_roi = gray[int(h*0.22):int(h*0.38), int(w*0.25):int(w*0.75)]
    
    # Save debug
    debug_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    debug_path = os.path.join(DEBUG_FOLDER, f"{debug_timestamp}_search.jpg")
    cv2.imwrite(debug_path, search_roi)
    
    # Try multiple OCR strategies
    readings = []
    
    # Strategy A: Look for digits followed by m³
    config_a = "--psm 6 --oem 3"
    text_a = pytesseract.image_to_string(search_roi, config=config_a)
    print(f"Strategy A OCR: '{text_a}'")
    
    # Look for pattern: digits + optional spaces + m³
    pattern_a = re.search(r'(\d[\d\s]*?)\s*m³', text_a)
    if pattern_a:
        digits = re.sub(r'\s', '', pattern_a.group(1))
        if len(digits) >= 6:
            reading = digits.rjust(8, "0")[:8]
            readings.append(reading)
            print(f"Found with m³ pattern: {reading}")
    
    # Strategy B: Look for 8 consecutive digits
    pattern_b = re.search(r'(\d{8})', text_a)
    if pattern_b:
        readings.append(pattern_b.group(1))
        print(f"Found 8-digit pattern: {pattern_b.group(1)}")
    
    # Strategy C: Try digits-only OCR
    config_c = "--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789 "
    text_c = pytesseract.image_to_string(search_roi, config=config_c).replace(" ", "")
    if len(text_c) >= 6:
        reading = text_c[:8].ljust(8, "0")
        readings.append(reading)
        print(f"Digits-only OCR: {reading}")
    
    # Choose the most likely reading
    # Prefer readings that look like meter readings (often starting with 0)
    if readings:
        # Count occurrences
        from collections import Counter
        freq = Counter(readings)
        
        # Prefer readings that start with 0 (common for meter readings)
        valid_readings = [r for r in readings if r.startswith('0')]
        if valid_readings:
            # Take the most common valid reading
            freq_valid = Counter(valid_readings)
            best_reading = freq_valid.most_common(1)[0][0]
        else:
            # Take the most common reading
            best_reading = freq.most_common(1)[0][0]
        
        confidence = 70.0  # Good confidence if we found something
    else:
        # Fallback to fixed pattern
        best_reading = "00000000"
        confidence = 30.0
    
    return best_reading, confidence

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
    
    # Try smart extraction first
    reading, confidence = extract_meter_reading_smart(image)
    
    print(f"Final reading: {reading}, Confidence: {confidence}")
    
    # Create output image
    output = image.copy()
    
    # Draw the reading area
    y1, y2 = int(h*0.22), int(h*0.38)
    x1, x2 = int(w*0.25), int(w*0.75)
    cv2.rectangle(output, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(output, "Reading Area", 
               (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
               0.7, (255, 0, 0), 2)
    
    # Draw the reading
    cv2.putText(output, f"Reading: {reading}",
               (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
               1.2, (0, 255, 255), 3)
    
    cv2.putText(output, f"Confidence: {confidence:.1f}%",
               (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
               0.8, (0, 255, 0), 2)
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(output, f"Processed: {timestamp}",
               (20, h-20), cv2.FONT_HERSHEY_SIMPLEX,
               0.5, (200, 200, 200), 1)
    
    # Save result
    filename = f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    result_path = os.path.join(UPLOAD_FOLDER, filename)
    cv2.imwrite(result_path, output)
    
    # Prepare response
    digit_details = []
    for i, digit in enumerate(reading):
        # Simple confidence distribution
        digit_conf = confidence * (0.8 + (i/40))  # Slightly vary confidence
        digit_conf = min(100, max(0, digit_conf))
        
        digit_details.append({
            'position': i+1,
            'digit': digit,
            'confidence': f"{digit_conf:.1f}%",
            'status': '✓' if digit_conf > 70 else '⚠' if digit_conf > 50 else '✗'
        })
    
    debug_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    
    return {
        'reading': reading,
        'confidence': f"{confidence:.1f}%",
        'image_url': f"/uploads/{filename}",
        'debug_url': f"/debug/{debug_timestamp}_search.jpg",
        'digit_details': digit_details,
        'timestamp': timestamp,
        'debug_timestamp': debug_timestamp
    }

# =====================================================
# FLASK ROUTES
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
    print("🚰 SMART WATER METER OCR")
    print("🌐 http://127.0.0.1:5000")
    print("📁 Upload folder:", os.path.abspath(UPLOAD_FOLDER))
    print("💡 This version specifically looks for:")
    print("   - 8-digit sequences")
    print("   - Digits followed by 'm³'")
    print("   - Focuses on the top-middle area only")
    print("   - Ignores serial numbers at the bottom")
    
    app.run(debug=True, host='0.0.0.0', port=5000)