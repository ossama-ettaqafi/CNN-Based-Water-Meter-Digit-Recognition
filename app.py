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
# PRECISE DIGIT DETECTION FOR WATER METERS
# =====================================================
def detect_precise_digits(image_roi, roi_coords, full_image):
    """Precisely detect individual digits in water meter display"""
    x1, y1, x2, y2 = roi_coords
    h_roi, w_roi = image_roi.shape
    
    # Save original ROI for debugging
    debug_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    debug_path = os.path.join(DEBUG_FOLDER, f"{debug_timestamp}_original_roi.jpg")
    cv2.imwrite(debug_path, image_roi)
    
    # Step 1: Enhance contrast specifically for black digits on white background
    # Water meters typically have high contrast black digits
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image_roi)
    
    # Step 2: Apply adaptive thresholding to isolate digits
    # Using Gaussian adaptive threshold works well for meter displays
    binary = cv2.adaptiveThreshold(enhanced, 255, 
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    # Save binary image for debugging
    binary_debug_path = os.path.join(DEBUG_FOLDER, f"{debug_timestamp}_binary.jpg")
    cv2.imwrite(binary_debug_path, binary)
    
    # Step 3: Clean up the binary image
    # Remove small noise
    kernel_clean = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_clean)
    
    # Step 4: Find contours - these should be our digits
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    digit_info = []
    min_digit_height = h_roi * 0.4  # Digits should be relatively tall
    max_digit_height = h_roi * 0.9  # But not too tall
    min_digit_width = w_roi * 0.05  # Minimum width for a digit
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        
        # Filter for digit-like characteristics:
        # 1. Height criteria (digits are tall relative to ROI)
        # 2. Width criteria 
        # 3. Area criteria (not too small)
        # 4. Aspect ratio (digits are usually taller than wide)
        if (min_digit_height < h < max_digit_height and
            w > min_digit_width and
            area > 100 and  # Minimum area
            0.3 < w/h < 1.0):  # Aspect ratio for digits
            
            # Calculate compactness (digits are less compact than other symbols)
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                compactness = 4 * np.pi * area / (perimeter * perimeter)
                # Digits typically have lower compactness (0.3-0.7)
                if 0.2 < compactness < 0.8:
                    
                    # Add small padding (2 pixels) for better OCR
                    pad = 2
                    x_start = max(0, x - pad)
                    y_start = max(0, y - pad)
                    x_end = min(w_roi, x + w + pad)
                    y_end = min(h_roi, y + h + pad)
                    
                    # Extract the digit region from the enhanced image
                    digit_roi = enhanced[y_start:y_end, x_start:x_end]
                    
                    # Resize to consistent size for better OCR
                    if digit_roi.size > 0:
                        target_height = 60
                        aspect_ratio = w / h
                        target_width = int(target_height * aspect_ratio)
                        digit_roi_resized = cv2.resize(digit_roi, 
                                                      (max(20, target_width), target_height),
                                                      interpolation=cv2.INTER_CUBIC)
                        
                        # Apply slight Gaussian blur to reduce noise
                        digit_roi_resized = cv2.GaussianBlur(digit_roi_resized, (1, 1), 0)
                        
                        # Binarize specifically for OCR
                        _, digit_binary = cv2.threshold(digit_roi_resized, 0, 255, 
                                                       cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        
                        # OCR on the individual digit
                        config_digit = "--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789"
                        digit_text = pytesseract.image_to_string(digit_binary, 
                                                                config=config_digit).strip()
                        
                        # If OCR fails, try with the resized but non-binary image
                        if not digit_text.isdigit():
                            digit_text = pytesseract.image_to_string(digit_roi_resized, 
                                                                    config=config_digit).strip()
                        
                        if digit_text and digit_text.isdigit():
                            digit = digit_text[0]
                            
                            # Calculate confidence based on contour properties
                            aspect_score = 1.0 - abs(0.6 - w/h)  # Ideal aspect ~0.6
                            height_score = 1.0 - abs(0.6 - h/h_roi)  # Ideal height ~60% of ROI
                            confidence = min(95.0, 70.0 + (aspect_score + height_score) * 12.5)
                            
                            # Calculate absolute coordinates
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
                                'contour_area': area,
                                'aspect_ratio': w/h
                            })
                            
                            # Save debug image for this digit
                            digit_debug_path = os.path.join(DEBUG_FOLDER, 
                                                          f"{debug_timestamp}_digit_{len(digit_info)}.jpg")
                            cv2.imwrite(digit_debug_path, digit_roi_resized)
    
    # Sort digits by their horizontal position (left to right)
    digit_info.sort(key=lambda d: d['center_x'])
    
    # Filter out overlapping/duplicate detections
    filtered_digits = []
    if digit_info:
        filtered_digits.append(digit_info[0])
        for i in range(1, len(digit_info)):
            current = digit_info[i]
            previous = filtered_digits[-1]
            
            # Check if this is overlapping with previous (within 30% of width)
            overlap_threshold = previous['w'] * 0.3
            if current['center_x'] - previous['center_x'] > overlap_threshold:
                filtered_digits.append(current)
            elif current['confidence'] > previous['confidence']:
                # If overlapping, keep the one with higher confidence
                filtered_digits[-1] = current
    
    print(f"Detected {len(filtered_digits)} precise digit(s)")
    return filtered_digits, debug_timestamp

def draw_precise_boxes(full_image, digit_info, reading, roi_coords=None):
    """Draw precise green boxes around detected digits"""
    output = full_image.copy()
    
    # Draw ROI if provided
    if roi_coords:
        x1, y1, x2, y2 = roi_coords
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 100, 255), 2)  # Orange for ROI
        cv2.putText(output, "Digit Search Area", 
                   (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (0, 100, 255), 2)
    
    # Draw precise green boxes for each detected digit
    for i, digit_data in enumerate(digit_info):
        x, y, w, h = digit_data['x'], digit_data['y'], digit_data['w'], digit_data['h']
        
        # Draw green rectangle with slightly thicker line for visibility
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Draw a filled background for the label
        label = f"{digit_data['digit']}"
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        cv2.rectangle(output, 
                     (x, y - text_size[1] - 10),
                     (x + text_size[0] + 10, y),
                     (0, 255, 0), -1)
        
        # Label with digit value in black
        cv2.putText(output, label,
                   (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                   0.8, (0, 0, 0), 2)
        
        # Add confidence below the box
        conf_text = f"{digit_data['confidence']:.0f}%"
        cv2.putText(output, conf_text,
                   (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, (0, 200, 0), 1)
    
    # Draw the complete reading at the top
    if reading:
        # Background for reading
        text_size = cv2.getTextSize(f"Reading: {reading}", cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        cv2.rectangle(output, 
                     (20, 20),
                     (20 + text_size[0] + 20, 20 + text_size[1] + 20),
                     (0, 0, 0), -1)
        
        cv2.putText(output, f"Reading: {reading}",
                   (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                   1.2, (0, 255, 255), 3)
    
    return output

# =====================================================
# PRECISE METER READING EXTRACTION
# =====================================================
def extract_precise_meter_reading(image):
    """Extract meter reading with precise digit detection"""
    h, w = image.shape[:2]
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Based on your image, the reading is in a very specific area:
    # Between "FERRO®" and the "m³" symbol
    # Let's define a very precise ROI
    
    # For water meters like in your image:
    roi_y1, roi_y2 = int(h * 0.28), int(h * 0.38)  # Very narrow vertical band
    roi_x1, roi_x2 = int(w * 0.35), int(w * 0.65)  # Centered horizontally
    
    # Extract ROI
    roi = gray[roi_y1:roi_y2, roi_x1:roi_x2]
    roi_coords = (roi_x1, roi_y1, roi_x2, roi_y2)
    
    # Detect individual digits precisely
    digit_info, debug_timestamp = detect_precise_digits(roi, roi_coords, image)
    
    # Construct reading from detected digits
    detected_digits = ''.join([d['digit'] for d in digit_info if d['digit'].isdigit()])
    
    # Try to get the complete reading from OCR as well
    config_full = "--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789"
    digits_text = pytesseract.image_to_string(roi, config=config_full).replace(" ", "")
    
    # Determine the best reading
    reading = ""
    confidence = 0.0
    
    if len(digit_info) >= 6:  # If we detected at least 6 digits
        # Use detected digits, pad to 8 digits
        reading = detected_digits[:8].ljust(8, "0")
        
        # Calculate average confidence of detected digits
        if digit_info:
            avg_confidence = sum(d['confidence'] for d in digit_info) / len(digit_info)
            confidence = min(95.0, avg_confidence * 0.9)
        else:
            confidence = 75.0
            
        print(f"Using {len(digit_info)} detected digits: {reading}")
        
    elif len(digits_text) >= 6:
        # Use OCR result
        reading = digits_text[:8].ljust(8, "0")
        confidence = 70.0
        print(f"Using OCR text: {reading}")
        
        # Try to match detected digits with OCR result
        if detected_digits and len(detected_digits) >= 4:
            # If OCR and detection mostly agree, increase confidence
            if detected_digits in reading or reading in detected_digits:
                confidence = 85.0
    else:
        # Try alternative OCR configuration
        config_alt = "--psm 8 --oem 3"
        alt_text = pytesseract.image_to_string(roi, config=config_alt)
        
        # Look for any 8-digit pattern
        match = re.search(r'(\d{8})', alt_text)
        if match:
            reading = match.group(1)
            confidence = 65.0
            print(f"Found 8-digit pattern: {reading}")
        else:
            # Look for any sequence of digits
            all_digits = re.findall(r'\d', alt_text)
            if len(all_digits) >= 6:
                reading = ''.join(all_digits[:8]).ljust(8, "0")
                confidence = 60.0
                print(f"Extracted from text: {reading}")
            else:
                reading = "00000000"
                confidence = 30.0
                print(f"No reliable reading found")
    
    # Create visualization with precise boxes
    output_image = draw_precise_boxes(image, digit_info, reading, roi_coords)
    
    # Add confidence and info text
    cv2.putText(output_image, f"Confidence: {confidence:.1f}%",
               (20, 90), cv2.FONT_HERSHEY_SIMPLEX,
               0.8, (0, 255, 0), 2)
    
    cv2.putText(output_image, f"Digits Detected: {len(digit_info)}",
               (20, 120), cv2.FONT_HERSHEY_SIMPLEX,
               0.7, (0, 200, 255), 2)
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(output_image, f"Processed: {timestamp}",
               (20, h-20), cv2.FONT_HERSHEY_SIMPLEX,
               0.5, (200, 200, 200), 1)
    
    return reading, confidence, output_image, digit_info, debug_timestamp

# =====================================================
# MAIN PROCESSING
# =====================================================
def process_image(image_path):
    """Main processing function with precise digit detection"""
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not read image")
    
    h, w = image.shape[:2]
    
    # Extract reading with precise digit detection
    reading, confidence, output_image, digit_info, debug_timestamp = extract_precise_meter_reading(image)
    
    print(f"Final reading: {reading}, Confidence: {confidence:.1f}%")
    
    # Save result image
    filename = f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    result_path = os.path.join(UPLOAD_FOLDER, filename)
    cv2.imwrite(result_path, output_image)
    
    # Prepare detailed digit information for response
    digit_details = []
    reading_digits = list(reading)
    
    for i, expected_digit in enumerate(reading_digits):
        detected = False
        digit_confidence = 0.0
        box_info = None
        
        # Try to find a detected digit at this position
        if i < len(digit_info):
            detected_digit = digit_info[i]
            detected = (detected_digit['digit'] == expected_digit)
            digit_confidence = detected_digit['confidence']
            box_info = {
                'x': detected_digit['x'],
                'y': detected_digit['y'],
                'width': detected_digit['w'],
                'height': detected_digit['h']
            }
        else:
            # Estimate confidence based on position and overall confidence
            digit_confidence = confidence * (0.9 - (i * 0.05))
        
        status = '✓' if detected else '⚠' if digit_confidence > 60 else '✗'
        
        digit_details.append({
            'position': i + 1,
            'digit': expected_digit,
            'confidence': f"{digit_confidence:.1f}%",
            'detected': detected,
            'status': status,
            'box_coordinates': box_info
        })
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    return {
        'reading': reading,
        'confidence': f"{confidence:.1f}%",
        'image_url': f"/uploads/{filename}",
        'debug_url': f"/debug/{debug_timestamp}_original_roi.jpg",
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
    print("🚰 PRECISE WATER METER OCR WITH DIGIT DETECTION")
    print("🌐 http://127.0.0.1:5000")
    print("📁 Upload folder:", os.path.abspath(UPLOAD_FOLDER))
    print("🎯 Features for precision:")
    print("   - Narrow, targeted ROI for digit area")
    print("   - Height-based filtering (digits must be tall)")
    print("   - Aspect ratio filtering (digit-like shapes only)")
    print("   - Compactness filtering (excludes symbols like #, m³)")
    print("   - Overlap removal for clean detection")
    print("   - Individual digit OCR with confidence scoring")
    
    app.run(debug=True, host='0.0.0.0', port=5000)