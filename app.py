from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import numpy as np
import os
from datetime import datetime
from keras.models import load_model

app = Flask(__name__)

# Create upload folder
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

# Create digit templates for template matching fallback
def create_digit_templates():
    """Create simple digit templates for template matching"""
    templates = {}
    
    # Create simple 7-segment like digit templates (10x14)
    for digit in range(10):
        template = np.zeros((20, 14), dtype=np.uint8)
        
        if digit == 0:
            cv2.rectangle(template, (2, 2), (12, 18), 255, 1)
            cv2.rectangle(template, (3, 3), (11, 17), 255, -1)
        elif digit == 1:
            cv2.line(template, (7, 2), (7, 18), 255, 3)
        elif digit == 2:
            # Top
            cv2.line(template, (3, 3), (11, 3), 255, 2)
            # Top right
            cv2.line(template, (11, 3), (11, 9), 255, 2)
            # Middle
            cv2.line(template, (3, 9), (11, 9), 255, 2)
            # Bottom left
            cv2.line(template, (3, 9), (3, 15), 255, 2)
            # Bottom
            cv2.line(template, (3, 15), (11, 15), 255, 2)
        elif digit == 3:
            cv2.line(template, (3, 3), (11, 3), 255, 2)
            cv2.line(template, (11, 3), (11, 9), 255, 2)
            cv2.line(template, (3, 9), (11, 9), 255, 2)
            cv2.line(template, (11, 9), (11, 15), 255, 2)
            cv2.line(template, (3, 15), (11, 15), 255, 2)
        elif digit == 4:
            cv2.line(template, (3, 3), (3, 9), 255, 2)
            cv2.line(template, (3, 9), (11, 9), 255, 2)
            cv2.line(template, (11, 3), (11, 15), 255, 2)
        elif digit == 5:
            cv2.line(template, (3, 3), (11, 3), 255, 2)
            cv2.line(template, (3, 3), (3, 9), 255, 2)
            cv2.line(template, (3, 9), (11, 9), 255, 2)
            cv2.line(template, (11, 9), (11, 15), 255, 2)
            cv2.line(template, (3, 15), (11, 15), 255, 2)
        elif digit == 6:
            cv2.line(template, (3, 3), (11, 3), 255, 2)
            cv2.line(template, (3, 3), (3, 15), 255, 2)
            cv2.line(template, (3, 9), (11, 9), 255, 2)
            cv2.line(template, (11, 9), (11, 15), 255, 2)
            cv2.line(template, (3, 15), (11, 15), 255, 2)
        elif digit == 7:
            cv2.line(template, (3, 3), (11, 3), 255, 2)
            cv2.line(template, (11, 3), (7, 15), 255, 2)
        elif digit == 8:
            cv2.rectangle(template, (2, 2), (12, 18), 255, 1)
            cv2.rectangle(template, (3, 3), (11, 17), 255, -1)
            cv2.line(template, (3, 9), (11, 9), 0, 2)
        elif digit == 9:
            cv2.rectangle(template, (2, 2), (12, 18), 255, 1)
            cv2.rectangle(template, (3, 3), (11, 17), 255, -1)
            cv2.rectangle(template, (3, 10), (11, 17), 0, -1)
        
        templates[digit] = template
    
    return templates

# Create templates
DIGIT_TEMPLATES = create_digit_templates()

def predict_digit_template_matching(digit_img):
    """Predict digit using template matching as fallback"""
    if len(digit_img.shape) == 3:
        digit_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
    
    # Resize to match template size
    resized = cv2.resize(digit_img, (14, 20))
    
    # Threshold
    _, binary = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    best_match = 0
    best_score = -1
    
    for digit, template in DIGIT_TEMPLATES.items():
        # Resize template to match digit size
        resized_template = cv2.resize(template, (binary.shape[1], binary.shape[0]))
        
        # Calculate match score
        score = np.sum(binary == resized_template) / binary.size
        
        if score > best_score:
            best_score = score
            best_match = digit
    
    return best_match, best_score

def predict_digit_combined(digit_img):
    """Try CNN first, fall back to template matching"""
    # Try CNN first
    if cnn_model is not None:
        try:
            # Prepare for CNN
            if len(digit_img.shape) == 2:
                cnn_input = cv2.cvtColor(digit_img, cv2.COLOR_GRAY2RGB)
            else:
                cnn_input = digit_img.copy()
            
            cnn_input = cv2.resize(cnn_input, (20, 32))
            cnn_input = cnn_input.astype('float32') / 255.0
            cnn_input = np.expand_dims(cnn_input, axis=0)
            
            probs = cnn_model.predict(cnn_input, verbose=0)
            cnn_digit = np.argmax(probs)
            cnn_confidence = float(np.max(probs))
            
            # If CNN confidence is high enough, use it
            if cnn_confidence > 0.7:
                return cnn_digit, cnn_confidence
        except:
            pass
    
    # Fall back to template matching
    return predict_digit_template_matching(digit_img)

def preprocess_digit_ferro_style(digit_img):
    """Preprocess Ferro meter digit specifically"""
    if len(digit_img.shape) == 3:
        digit_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
    
    # 1. Resize to reasonable size
    height, width = digit_img.shape
    if height == 0 or width == 0:
        return None
    
    # 2. Apply CLAHE for contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(digit_img)
    
    # 3. Apply Gaussian blur
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    # 4. Threshold - Ferro digits are usually dark on light background
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 5. Invert if needed (we want white digit on black background)
    # Check if most pixels are black (digit should be white)
    white_pixels = np.sum(binary == 255)
    black_pixels = np.sum(binary == 0)
    
    if white_pixels > black_pixels:
        # Invert so digit is white on black
        binary = cv2.bitwise_not(binary)
    
    # 6. Clean up small noise
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # 7. Find the largest contour (the digit)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Extract the digit
        digit_roi = cleaned[y:y+h, x:x+w]
        
        # Add padding
        padding = 5
        digit_padded = cv2.copyMakeBorder(digit_roi, padding, padding, padding, padding,
                                         cv2.BORDER_CONSTANT, value=0)
    else:
        digit_padded = cleaned
    
    # 8. Resize to standard size
    resized = cv2.resize(digit_padded, (20, 32))
    
    # Create display version (inverted for visualization)
    display_version = cv2.bitwise_not(resized)
    
    return resized, display_version

def detect_and_extract_precise(roi_gray, output_image, roi_coords):
    """Detect and extract digits with precise segmentation"""
    x1, y1, x2, y2 = roi_coords
    h_roi, w_roi = roi_gray.shape
    
    # DEBUG: Save the ROI for inspection
    debug_roi = roi_gray.copy()
    
    # 1. Apply strong preprocessing to enhance digits
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(roi_gray)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    # Threshold to get binary image
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # DEBUG: Save binary image
    debug_binary = binary.copy()
    
    # 2. Find contours of all potential digits
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 3. Filter contours to get only digit-like shapes
    digit_contours = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # Calculate features
        area = w * h
        aspect_ratio = h / w if w > 0 else 0
        contour_area = cv2.contourArea(contour)
        solidity = contour_area / area if area > 0 else 0
        
        # Ferro digit characteristics:
        # - Not too small (at least 30 pixels)
        # - Not too large (less than 30% of ROI area)
        # - Aspect ratio typically between 1.2 and 3.0
        # - Reasonable solidity (not too fragmented)
        if (area > 30 and 
            area < h_roi * w_roi * 0.3 and
            1.0 < aspect_ratio < 4.0 and
            solidity > 0.2):
            
            digit_contours.append({
                'x': x, 'y': y, 'w': w, 'h': h,
                'area': area,
                'aspect_ratio': aspect_ratio,
                'center_x': x + w//2,
                'center_y': y + h//2
            })
    
    # 4. Sort contours by x-position
    digit_contours.sort(key=lambda c: c['x'])
    
    # 5. Group contours that are likely part of the same digit line
    if digit_contours:
        # Calculate median y-position
        y_positions = [c['center_y'] for c in digit_contours]
        median_y = np.median(y_positions)
        
        # Select contours close to median y (within 10 pixels)
        selected_contours = [c for c in digit_contours 
                           if abs(c['center_y'] - median_y) < 15]
        
        # If we have too few, relax the constraint
        if len(selected_contours) < 4:
            selected_contours = digit_contours
    
    # 6. Sort selected contours by x-position
    selected_contours.sort(key=lambda c: c['x'])
    
    # 7. Limit to 8 contours (Ferro has 8 digits)
    if len(selected_contours) > 8:
        # Take the 8 with largest area (most likely to be digits)
        selected_contours.sort(key=lambda c: c['area'], reverse=True)
        selected_contours = selected_contours[:8]
        selected_contours.sort(key=lambda c: c['x'])
    
    result = ""
    confidences = []
    digit_images = []
    
    # 8. Process each digit contour
    for i, contour_info in enumerate(selected_contours):
        x = contour_info['x']
        y = contour_info['y']
        w = contour_info['w']
        h = contour_info['h']
        
        # Extract digit from original grayscale
        digit_img = roi_gray[y:y+h, x:x+w]
        
        if digit_img.size < 20:  # Too small
            continue
        
        # Preprocess digit
        processed = preprocess_digit_ferro_style(digit_img)
        if processed is None:
            continue
        
        processed_digit, display_digit = processed
        
        # Predict using combined method
        digit_val, confidence = predict_digit_combined(processed_digit)
        
        # Special handling for Ferro digits:
        # Ferro "0" often looks like "8" to simple recognizers
        # Ferro "2" has specific shape
        
        # Apply some heuristics based on contour shape
        aspect_ratio = h / w if w > 0 else 0
        
        # If aspect ratio is very high (tall and thin), might be "1"
        if aspect_ratio > 2.5 and confidence < 0.8:
            digit_val = 1
            confidence = max(confidence, 0.7)
        
        result += str(digit_val)
        confidences.append(confidence)
        digit_images.append((display_digit, digit_val, confidence))
        
        # Draw on output
        abs_x = x1 + x
        abs_y = y1 + y
        
        # Draw bounding box
        color = (0, 255, 0)  # Green
        cv2.rectangle(output_image, 
                     (abs_x, abs_y), 
                     (abs_x + w, abs_y + h), 
                     color, 2)
        
        # Draw prediction
        cv2.putText(output_image, f"{digit_val}", 
                   (abs_x + w//2 - 5, abs_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw confidence
        cv2.putText(output_image, f"{confidence:.0%}", 
                   (abs_x, abs_y + h + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1)
    
    # 9. If we have too few digits, try alternative method
    if len(result) < 5:
        print(f"Only found {len(result)} digits, trying alternative method...")
        
        # Try simple vertical projection
        vertical_proj = np.sum(binary, axis=0)
        
        # Find regions with high projection
        threshold = np.max(vertical_proj) * 0.2
        active_indices = np.where(vertical_proj > threshold)[0]
        
        if len(active_indices) > 0:
            # Group indices into digit regions
            regions = []
            current_start = active_indices[0]
            current_end = active_indices[0]
            
            for i in range(1, len(active_indices)):
                if active_indices[i] - active_indices[i-1] > 5:
                    if current_end - current_start > 3:
                        regions.append((current_start, current_end))
                    current_start = active_indices[i]
                current_end = active_indices[i]
            
            if current_end - current_start > 3:
                regions.append((current_start, current_end))
            
            regions.sort(key=lambda r: r[0])
            
            # Process each region
            for start_x, end_x in regions[:8]:
                width = end_x - start_x
                padding = width // 3
                
                x_start = max(0, start_x - padding)
                x_end = min(w_roi, end_x + padding)
                
                # Extract region
                region_img = roi_gray[:, x_start:x_end]
                
                if region_img.size < 50:
                    continue
                
                # Preprocess and predict
                processed = preprocess_digit_ferro_style(region_img)
                if processed is None:
                    continue
                
                processed_digit, display_digit = processed
                digit_val, confidence = predict_digit_combined(processed_digit)
                
                result += str(digit_val)
                confidences.append(confidence)
                digit_images.append((display_digit, digit_val, confidence))
                
                # Draw in cyan for fallback
                abs_x = x1 + x_start
                abs_y = y1
                w = x_end - x_start
                h = h_roi
                
                cv2.rectangle(output_image, 
                            (abs_x, abs_y), 
                            (abs_x + w, abs_y + h), 
                            (255, 255, 0), 1)  # Cyan
                
                cv2.putText(output_image, f"{digit_val}", 
                           (abs_x + w//2 - 5, abs_y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
    
    return result, confidences, digit_images, output_image

def create_detailed_strip(digit_images, output_image):
    """Create detailed visualization strip"""
    if not digit_images:
        return output_image
    
    strip_height = 80
    num_digits = len(digit_images)
    strip_width = max(300, num_digits * 25)
    
    # Create strip
    strip = np.ones((strip_height, strip_width), dtype=np.uint8) * 255
    
    # Add title
    strip_bgr = cv2.cvtColor(strip, cv2.COLOR_GRAY2BGR)
    cv2.putText(strip_bgr, "EXTRACTED DIGITS (Ferro Meter):", 
               (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # Add each digit
    for i, (digit_img, digit_val, confidence) in enumerate(digit_images):
        if digit_img is not None and digit_img.size > 0:
            try:
                # Resize for display
                display_resized = cv2.resize(digit_img, (20, 40))
                
                # Place in strip
                y_offset = 25
                x_pos = i * 25 + 10
                
                # Convert to 3 channels
                if len(display_resized.shape) == 2:
                    display_bgr = cv2.cvtColor(display_resized, cv2.COLOR_GRAY2BGR)
                else:
                    display_bgr = display_resized
                
                # Place in strip
                strip_bgr[y_offset:y_offset+40, x_pos:x_pos+20] = display_bgr
                
                # Add border
                cv2.rectangle(strip_bgr, 
                            (x_pos, y_offset), 
                            (x_pos+20, y_offset+40), 
                            (0, 0, 0), 1)
                
                # Add digit value
                cv2.putText(strip_bgr, f"={digit_val}", 
                           (x_pos + 22, y_offset + 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 0), 1)
                
                # Add confidence
                cv2.putText(strip_bgr, f"{confidence:.0%}", 
                           (x_pos, y_offset + 55),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 150), 1)
                
            except Exception as e:
                print(f"Error displaying digit {i}: {e}")
    
    # Resize to match output width
    output_width = output_image.shape[1]
    strip_resized = cv2.resize(strip_bgr, (output_width, strip_height))
    
    # Add to output
    return np.vstack([output_image, strip_resized])

def process_image(image_path):
    """Main processing pipeline"""
    image = cv2.imread(image_path)
    if image is None:
        return "Error: Could not read image", 0.0, None
    
    output = image.copy()
    
    # Use the working detection method from earlier
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h_img, w_img = gray.shape
    
    # Original detection method that worked
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    row_sums = np.sum(thresh, axis=1)
    upper_limit = int(h_img * 0.2)
    lower_limit = int(h_img * 0.8)
    relevant_rows = row_sums[upper_limit:lower_limit]
    
    if len(relevant_rows) > 0:
        peak_y = np.argmax(relevant_rows) + upper_limit
    else:
        peak_y = h_img // 2
    
    roi_h = int(h_img * 0.10)
    roi_w = int(w_img * 0.6)
    
    y1 = max(0, peak_y - roi_h // 2)
    y2 = min(h_img, peak_y + roi_h // 2)
    x1 = (w_img // 2) - (roi_w // 2)
    x2 = (w_img // 2) + (roi_w // 2)
    
    # Draw register box
    cv2.rectangle(output, (x1, y1), (x2, y2), (255, 0, 0), 3)
    cv2.putText(output, "Ferro Digit Register", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # Extract ROI
    roi_color = image[y1:y2, x1:x2]
    if roi_color.size == 0:
        return "Error: Invalid ROI", 0.0, None
    
    roi_gray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
    
    # Extract digits
    result, confidences, digit_images, output_with_digits = detect_and_extract_precise(
        roi_gray, output, (x1, y1, x2, y2))
    
    # Add visualization strip
    output_with_strip = create_detailed_strip(digit_images, output_with_digits)
    
    # Add result overlay
    if result:
        # Ensure we have 8 digits
        if len(result) < 8:
            result = result.ljust(8, '0')
        elif len(result) > 8:
            result = result[:8]
        
        avg_conf = np.mean(confidences) * 100 if confidences else 0
        
        # Create overlay
        overlay = output_with_strip.copy()
        text_bg_height = 90 if len(confidences) > 0 else 60
        cv2.rectangle(overlay, (10, 10), (400, text_bg_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, output_with_strip, 0.3, 0, output_with_strip)
        
        # Add reading
        reading_text = f"FERRO METER: {result}"
        cv2.putText(output_with_strip, reading_text, 
                   (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        
        # Add confidence
        if confidences:
            conf_text = f"Confidence: {avg_conf:.1f}% ({len(confidences)} digits)"
            cv2.putText(output_with_strip, conf_text, 
                       (20, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Add large result
        cv2.putText(output_with_strip, result, 
                   (output_with_strip.shape[1] - 200, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4)
    else:
        avg_conf = 0.0
        result = "00000000"
        cv2.putText(output_with_strip, "No Ferro digits detected", 
                   (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Save result
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_filename = f"result_{timestamp}.jpg"
    result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
    
    cv2.imwrite(result_path, output_with_strip)
    
    return result, avg_conf, result_filename

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Save original
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    original_filename = f"original_{timestamp}_{file.filename}"
    original_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
    file.save(original_path)
    
    # Process image
    try:
        reading, confidence, result_filename = process_image(original_path)
        
        return jsonify({
            'success': True,
            'reading': reading,
            'confidence': f"{confidence:.1f}%",
            'image_url': f"/uploads/{result_filename}" if result_filename else None,
            'digit_count': len(reading)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'reading': "00000000",
            'confidence': "0.0%"
        }), 500

if __name__ == '__main__':
    print("=" * 60)
    print("🔢 FERRO METER DIGIT RECOGNITION - IMPROVED")
    print("=" * 60)
    print(f"📁 Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f"🤖 CNN Model: {'✅ Loaded' if cnn_model else '❌ Not loaded'}")
    print(f"🔧 Template Matching: ✅ Ready")
    print(f"🌐 Server: http://127.0.0.1:5000")
    print("=" * 60)
    print("📸 Upload Ferro meter images to extract 8-digit readings!")
    print("=" * 60)
    
    app.run(debug=True, port=5000)