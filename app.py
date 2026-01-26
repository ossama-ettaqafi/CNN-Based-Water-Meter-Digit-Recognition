from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import numpy as np
import os
from datetime import datetime
import pytesseract

# ================== TESSERACT CONFIG ==================
pytesseract.pytesseract.tesseract_cmd = r"D:\Program Files\Tesseract-OCR\tesseract.exe"

# ================== FLASK APP ==================
app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# =====================================================
# OCR DIGIT RECOGNITION
# =====================================================
def ocr_digit(img):
    if img is None or img.size == 0:
        return "0", 0.0

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.resize(img, (40, 64), interpolation=cv2.INTER_CUBIC)

    _, img = cv2.threshold(img, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    config = "--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789"
    text = pytesseract.image_to_string(img, config=config).strip()

    if text.isdigit():
        return text, 1.0
    return "0", 0.0

# =====================================================
# DIGIT PREPROCESSING
# =====================================================
def preprocess_digit_for_ocr(digit_img):
    if len(digit_img.shape) == 3:
        digit_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(2.0, (4, 4))
    enhanced = clahe.apply(digit_img)

    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    _, binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        digit = binary[y:y+h, x:x+w]
        digit = cv2.copyMakeBorder(digit, 5, 5, 5, 5,
                                   cv2.BORDER_CONSTANT, value=0)
    else:
        digit = binary

    digit = cv2.resize(digit, (40, 64))
    digit = cv2.bitwise_not(digit)
    return digit

# =====================================================
# REGISTER DETECTION
# =====================================================
def detect_register_precise(image):
    h, w = image.shape[:2]
    x1 = int(w * 0.2)
    x2 = int(w * 0.8)
    y1 = int(h * 0.22)
    y2 = int(h * 0.35)
    return x1, y1, x2, y2

# =====================================================
# DIGIT EXTRACTION (VERTICAL PROJECTION)
# =====================================================
def extract_digits_vertical_projection(roi_gray, output, roi_coords):
    x1, y1, x2, y2 = roi_coords
    h, w = roi_gray.shape

    clahe = cv2.createCLAHE(3.0, (8, 8))
    enhanced = clahe.apply(roi_gray)
    _, binary = cv2.threshold(
        enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    projection = np.sum(binary, axis=0)
    projection = projection / np.max(projection)

    regions = []
    in_digit = False
    start = 0

    for i, v in enumerate(projection):
        if v > 0.15 and not in_digit:
            in_digit = True
            start = i
        elif v <= 0.15 and in_digit:
            if i - start > 5:
                regions.append((start, i))
            in_digit = False

    regions = regions[:8]

    result = ""
    confidences = []

    for start_x, end_x in regions:
        digit_col = binary[:, start_x:end_x]
        rows = np.where(np.sum(digit_col, axis=1) > 0)[0]

        if len(rows) == 0:
            continue

        y_start = max(0, rows[0] - 5)
        y_end = min(h, rows[-1] + 5)

        digit_img = roi_gray[y_start:y_end, start_x:end_x]
        digit_img = preprocess_digit_for_ocr(digit_img)

        digit, conf = ocr_digit(digit_img)
        result += digit
        confidences.append(conf)

        abs_x = x1 + start_x
        abs_y = y1 + y_start
        cv2.rectangle(output,
                      (abs_x, abs_y),
                      (abs_x + (end_x - start_x), abs_y + (y_end - y_start)),
                      (0, 255, 0), 2)
        cv2.putText(output, digit,
                    (abs_x, abs_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return result, confidences, output

# =====================================================
# MAIN PIPELINE
# =====================================================
def process_image(image_path):
    image = cv2.imread(image_path)
    output = image.copy()

    x1, y1, x2, y2 = detect_register_precise(image)
    cv2.rectangle(output, (x1, y1), (x2, y2), (255, 0, 0), 3)

    roi = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    result, confidences, output = extract_digits_vertical_projection(
        roi, output, (x1, y1, x2, y2)
    )

    if len(result) < 8:
        result = result.ljust(8, "0")
    result = result[:8]

    conf = np.mean(confidences) * 100 if confidences else 0

    cv2.putText(output, f"Meter Reading: {result}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 255, 255), 2)

    filename = f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    cv2.imwrite(os.path.join(UPLOAD_FOLDER, filename), output)

    return result, conf, filename

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
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No image"}), 400

    name = f"input_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    path = os.path.join(UPLOAD_FOLDER, name)
    file.save(path)

    reading, conf, out = process_image(path)

    return jsonify({
        "success": True,
        "reading": reading,
        "confidence": f"{conf:.1f}%",
        "image_url": f"/uploads/{out}"
    })

# =====================================================
if __name__ == "__main__":
    print("🚰 FERRO WATER METER OCR (NO CNN)")
    print("🌐 http://127.0.0.1:5000")
    app.run(debug=True)
