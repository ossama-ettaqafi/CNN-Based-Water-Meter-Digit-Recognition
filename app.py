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
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max file size

# =====================================================
# IMAGE PREPROCESSING ENHANCEMENTS
# =====================================================
def enhance_image_quality(image):
    """Enhance image quality for better OCR results"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Reduce noise while preserving edges
    denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    return denoised

def remove_grid_background(image):
    """Remove grid/background patterns to isolate meter digits"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply median blur to remove small grid patterns
    blurred = cv2.medianBlur(gray, 5)
    
    # Use morphological operations to remove grid lines
    kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    
    # Remove horizontal lines
    horizontal = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel_horizontal)
    removed_horizontal = cv2.subtract(blurred, horizontal)
    
    # Remove vertical lines
    vertical = cv2.morphologyEx(removed_horizontal, cv2.MORPH_OPEN, kernel_vertical)
    removed_both = cv2.subtract(removed_horizontal, vertical)
    
    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(removed_both)
    
    return enhanced

# =====================================================
# ADVANCED OCR DIGIT RECOGNITION
# =====================================================
def ocr_digit_advanced(img):
    """Advanced OCR with multiple fallback strategies"""
    if img is None or img.size == 0:
        return "0", 0.0
    
    # Ensure grayscale
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Resize to optimal size for Tesseract
    img = cv2.resize(img, (40, 64), interpolation=cv2.INTER_CUBIC)
    
    # Multiple thresholding strategies
    results = []
    
    # Strategy 1: Otsu's thresholding
    _, thresh1 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Strategy 2: Adaptive thresholding
    thresh2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)
    
    thresholds = [thresh1, thresh2]
    
    for i, thresh_img in enumerate(thresholds):
        # Try different PSM modes
        psm_modes = [10, 8]  # Single character, single word
        
        for psm in psm_modes:
            config = f"--psm {psm} --oem 3 -c tessedit_char_whitelist=0123456789"
            
            # Get detailed OCR data
            data = pytesseract.image_to_data(thresh_img, config=config, 
                                            output_type=pytesseract.Output.DICT)
            
            # Check each detected text
            for j in range(len(data['text'])):
                text = data['text'][j].strip()
                if j < len(data['conf']):
                    conf = float(data['conf'][j]) if data['conf'][j] != '' else 0.0
                else:
                    conf = 0.0
                
                if text.isdigit() and conf > 0:
                    # Validate digit shape (basic aspect ratio check)
                    h, w = thresh_img.shape
                    if 0.5 < h/w < 2.0:  # Reasonable aspect ratio for digits
                        results.append((text, conf, i, psm))
    
    # Sort by confidence
    results.sort(key=lambda x: x[1], reverse=True)
    
    if results:
        best_digit = results[0][0]
        best_conf = results[0][1]
        
        # Verify with majority voting among top results
        top_results = [r[0] for r in results[:3] if r[1] > 60]
        if len(top_results) >= 2:
            most_common = Counter(top_results).most_common(1)[0]
            if most_common[1] >= 2:  # At least 2 agree
                best_digit = most_common[0]
        
        return best_digit, best_conf
    
    return "0", 0.0

# =====================================================
# IMPROVED DIGIT PREPROCESSING
# =====================================================
def preprocess_digit_for_ocr(digit_img):
    """Enhanced digit preprocessing"""
    if len(digit_img.shape) == 3:
        digit_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
    
    # Resize to standard size first
    digit_img = cv2.resize(digit_img, (40, 64), interpolation=cv2.INTER_AREA)
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(digit_img)
    
    # Denoise
    denoised = cv2.medianBlur(enhanced, 3)
    
    # Threshold
    _, binary = cv2.threshold(denoised, 0, 255, 
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours and extract largest connected component
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, 
                                   cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get the largest contour
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        
        # Add some padding
        pad = 5
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(binary.shape[1] - x, w + 2*pad)
        h = min(binary.shape[0] - y, h + 2*pad)
        
        digit = binary[y:y+h, x:x+w]
    else:
        digit = binary
    
    # Resize back to standard size
    digit = cv2.resize(digit, (40, 64), interpolation=cv2.INTER_AREA)
    
    # Invert back (Tesseract expects black text on white background)
    digit = cv2.bitwise_not(digit)
    
    return digit

# =====================================================
# SMART METER DIGIT DETECTION
# =====================================================
def detect_meter_digits(image):
    """Smart detection specifically for meter display digits"""
    h, w = image.shape[:2]
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Remove grid background
    cleaned = remove_grid_background(gray)
    
    # Find bright areas (likely meter display)
    _, binary = cv2.threshold(cleaned, 200, 255, cv2.THRESH_BINARY)
    
    # Find contours of bright regions
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # Fallback to original detection
        return detect_register_precise(image)
    
    # Filter contours by size and aspect ratio (looking for digit-like regions)
    digit_contours = []
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        aspect_ratio = cw / ch
        
        # Look for regions that could be digits or digit groups
        if 0.2 < aspect_ratio < 5.0 and cw > w/20 and ch > h/20:
            digit_contours.append((x, y, cw, ch, cv2.contourArea(contour)))
    
    if digit_contours:
        # Find the largest bright region (likely the meter display)
        digit_contours.sort(key=lambda c: c[4], reverse=True)
        x, y, cw, ch, _ = digit_contours[0]
        
        # Expand region slightly
        expand_x = int(cw * 0.1)
        expand_y = int(ch * 0.1)
        x1 = max(0, x - expand_x)
        y1 = max(0, y - expand_y)
        x2 = min(w, x + cw + expand_x)
        y2 = min(h, y + ch + expand_y)
        
        return x1, y1, x2, y2
    else:
        # Fallback
        return detect_register_precise(image)

def detect_register_precise(image):
    """Detect meter reading area with adaptive positioning"""
    h, w = image.shape[:2]
    
    # Based on typical meter layout
    x1 = int(w * 0.25)  # More centered
    x2 = int(w * 0.75)
    y1 = int(h * 0.25)
    y2 = int(h * 0.40)
    
    return x1, y1, x2, y2

# =====================================================
# ENHANCED DIGIT EXTRACTION
# =====================================================
def extract_meter_digits(roi_gray, output, roi_coords):
    """Extract digits from meter display"""
    x1, y1, x2, y2 = roi_coords
    h, w = roi_gray.shape
    
    # Clean the ROI to remove background patterns
    cleaned_roi = remove_grid_background(roi_gray)
    
    # Enhance contrast
    enhanced = enhance_image_quality(cleaned_roi)
    
    # Use adaptive thresholding to handle varying lighting
    binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # Clean up noise
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Find connected components (digit candidates)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    digit_regions = []
    
    # Filter components by size and aspect ratio
    for i in range(1, num_labels):  # Skip background (0)
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        
        # Filter by size and aspect ratio (digit-like regions)
        aspect_ratio = width / height
        if (0.3 < aspect_ratio < 1.2 and  # Digit-like aspect ratio
            area > 50 and area < (w * h) / 10 and  # Reasonable size
            height > h * 0.3):  # Minimum height
            digit_regions.append((x, y, width, height, centroids[i][0]))
    
    # Sort by x-coordinate (left to right)
    digit_regions.sort(key=lambda r: r[4])
    
    # Limit to 8 digits max
    digit_regions = digit_regions[:8]
    
    # If we have fewer than 8 regions, use fixed positions
    if len(digit_regions) < 8:
        # Estimate digit positions
        digit_width = w // 9
        for i in range(8):
            x = i * digit_width
            digit_regions.append((x, 0, digit_width, h, x + digit_width/2))
        digit_regions = digit_regions[:8]
        digit_regions.sort(key=lambda r: r[4])
    
    result = ""
    confidences = []
    digit_positions = []
    
    for idx, (x, y, width, height, _) in enumerate(digit_regions):
        # Extract digit with padding
        pad_x = 3
        pad_y = 3
        x_start = max(0, x - pad_x)
        y_start = max(0, y - pad_y)
        x_end = min(w, x + width + pad_x)
        y_end = min(h, y + height + pad_y)
        
        digit_img = roi_gray[y_start:y_end, x_start:x_end]
        
        if digit_img.size == 0:
            result += "0"
            confidences.append(0.0)
            continue
        
        # Preprocess and OCR
        processed_digit = preprocess_digit_for_ocr(digit_img)
        digit, conf = ocr_digit_advanced(processed_digit)
        result += digit
        confidences.append(conf)
        
        # Store for visualization
        abs_x = x1 + x_start
        abs_y = y1 + y_start
        digit_positions.append((abs_x, abs_y, x_end - x_start, y_end - y_start, digit, conf))
        
        # Draw bounding box
        cv2.rectangle(output, 
                     (abs_x, abs_y),
                     (abs_x + (x_end - x_start), abs_y + (y_end - y_start)),
                     (0, 255, 0), 2)
        
        # Label with digit and confidence
        cv2.putText(output, f"{digit} ({conf:.0f}%)",
                   (abs_x, abs_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Draw ROI rectangle
    cv2.rectangle(output, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(output, "Meter Digits", 
               (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
               0.7, (255, 0, 0), 2)
    
    return result, confidences, output, digit_positions

# =====================================================
# ADDITIONAL METER DATA EXTRACTION
# =====================================================
def extract_additional_info(image):
    """Extract additional information from meter if available"""
    info = {}
    h, w = image.shape[:2]
    
    # Convert to grayscale for OCR
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Clean grid background first
    cleaned = remove_grid_background(gray)
    
    # Look for flow rate (Q) - typically in bottom area
    q_roi = cleaned[int(h*0.7):int(h*0.85), int(w*0.1):int(w*0.4)]
    q_text = pytesseract.image_to_string(q_roi, 
                                        config="--psm 6 --oem 3").strip()
    
    # Look for serial number or other identifiers
    serial_roi = cleaned[int(h*0.85):h, int(w*0.6):w]
    serial_text = pytesseract.image_to_string(serial_roi, 
                                             config="--psm 6 --oem 3").strip()
    
    # Clean extracted text
    q_match = re.search(r'Q[:\s]*([\d\.]+)\s*m³/h', q_text)
    if q_match:
        info['flow_rate'] = q_match.group(1)
    
    # Extract any numbers that look like serial numbers
    serial_match = re.search(r'(\d{6,})', serial_text)
    if serial_match:
        info['serial'] = serial_match.group(1)
    
    return info

# =====================================================
# MAIN PIPELINE WITH ENHANCEMENTS
# =====================================================
def process_image(image_path):
    """Main processing pipeline with improved accuracy"""
    # Read and validate image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not read image")
    
    # Get image dimensions
    h, w = image.shape[:2]
    
    # Create output image
    output = image.copy()
    
    # Detect meter digit area
    x1, y1, x2, y2 = detect_meter_digits(image)
    
    # Extract ROI
    roi = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    
    # Extract digits with enhanced method
    result, confidences, output_with_boxes, digit_positions = extract_meter_digits(
        roi, output, (x1, y1, x2, y2)
    )
    
    # Ensure 8-digit format
    if len(result) < 8:
        result = result.ljust(8, "0")
    result = result[:8]
    
    # Calculate overall confidence
    if confidences:
        valid_confidences = [c for c in confidences if c > 0]
        if valid_confidences:
            conf = np.mean(valid_confidences)
        else:
            conf = 0.0
    else:
        conf = 0.0
    
    # Extract additional information
    additional_info = extract_additional_info(image)
    
    # Add text overlay
    cv2.putText(output_with_boxes, f"Reading: {result}",
               (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
               1.2, (0, 255, 255), 3)
    
    cv2.putText(output_with_boxes, f"Confidence: {conf:.1f}%",
               (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
               0.8, (0, 255, 0), 2)
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(output_with_boxes, f"Processed: {timestamp}",
               (20, h-20), cv2.FONT_HERSHEY_SIMPLEX,
               0.5, (200, 200, 200), 1)
    
    # Save result
    filename = f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    result_path = os.path.join(UPLOAD_FOLDER, filename)
    cv2.imwrite(result_path, output_with_boxes)
    
    # Prepare detailed response
    digit_details = []
    for i, (digit, conf_digit) in enumerate(zip(result, confidences)):
        digit_details.append({
            'position': i+1,
            'digit': digit,
            'confidence': f"{conf_digit:.1f}%",
            'status': '✓' if conf_digit > 70 else '⚠' if conf_digit > 50 else '✗'
        })
    
    return {
        'reading': result,
        'confidence': f"{conf:.1f}%",
        'image_url': f"/uploads/{filename}",
        'digit_details': digit_details,
        'additional_info': additional_info,
        'timestamp': timestamp
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
        
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        app.logger.error(f"Processing error: {str(e)}")
        return jsonify({"error": "Internal processing error"}), 500

@app.route("/health")
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "Meter OCR"
    })

# =====================================================
if __name__ == "__main__":
    print("🚰 FERRO WATER METER OCR (GRID-FILTERED)")
    print("🌐 http://127.0.0.1:5000")
    print("📁 Upload folder:", os.path.abspath(UPLOAD_FOLDER))
    print("💡 Tips:")
    print("   - Ensure good lighting in meter photos")
    print("   - Keep camera parallel to meter surface")
    print("   - Avoid glare and reflections")
    print("   - Fill frame with meter display")
    
    app.run(debug=True, host='0.0.0.0', port=5000)