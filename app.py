import cv2
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt

# ----------------------------
# 1. Load model
# ----------------------------
model = load_model("model.keras", compile=False)

def predict_digit(img_digit):
    """Predict a single digit with preprocessing for RGB model"""
    # Ensure image is RGB
    if len(img_digit.shape) == 2:
        img_digit = cv2.cvtColor(img_digit, cv2.COLOR_GRAY2RGB)
    
    # Resize to model input
    img_digit = cv2.resize(img_digit, (20, 32))
    img_digit = img_digit.astype('float32') / 255.0
    img_digit = np.expand_dims(img_digit, axis=0)
    
    # Prediction
    probs = model.predict(img_digit, verbose=0)
    return np.argmax(probs), np.max(probs)

# ----------------------------
# 2. Load image
# ----------------------------
image_path = "images/image.png"
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Cannot load image: {image_path}")

output = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ----------------------------
# 3. Enhance contrast
# ----------------------------
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
gray = clahe.apply(gray)

# ----------------------------
# 4. Define ROI (adjust as needed)
# ----------------------------
h_img, w_img = gray.shape
roi_y, roi_h = int(h_img*0.35), int(h_img*0.12)
roi_x, roi_w = int(w_img*0.48), int(w_img*0.35)
roi = gray[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
roi_color = image[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]

# ----------------------------
# 5. Preprocess ROI for digits
# ----------------------------
_, roi_thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
kernel = np.ones((2,2), np.uint8)
roi_clean = cv2.morphologyEx(roi_thresh, cv2.MORPH_CLOSE, kernel)
roi_clean = cv2.morphologyEx(roi_clean, cv2.MORPH_OPEN, kernel)

# ----------------------------
# 6. Detect digit contours
# ----------------------------
contours, _ = cv2.findContours(roi_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
digit_contours = []
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    if h > roi_h * 0.4 and w > roi_w * 0.05 and w < roi_w * 0.2:
        digit_contours.append((x, y, w, h))

# Sort left to right
digit_contours.sort(key=lambda x: x[0])

if len(digit_contours) == 0:
    raise ValueError("No digits detected! Adjust ROI parameters.")

# Limit to max 8 digits
digit_contours = digit_contours[:8]

# ----------------------------
# 7. Predict each digit
# ----------------------------
compteur_final = ""
confidences = []

for x, y, w, h in digit_contours:
    digit_img = roi[y:y+h, x:x+w]

    # Threshold and invert if needed
    _, digit_processed = cv2.threshold(digit_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    margin = 5
    digit_with_margin = np.zeros((h+2*margin, w+2*margin), dtype=np.uint8)
    digit_with_margin[margin:margin+h, margin:margin+w] = digit_processed
    if np.mean(digit_with_margin) < 127:
        digit_with_margin = cv2.bitwise_not(digit_with_margin)

    # Convert to RGB
    digit_rgb = cv2.cvtColor(digit_with_margin, cv2.COLOR_GRAY2RGB)

    # Predict
    digit, confidence = predict_digit(digit_rgb)
    compteur_final += str(digit)
    confidences.append(confidence)

    # Draw rectangle and label
    cv2.rectangle(output, (roi_x+x, roi_y+y), (roi_x+x+w, roi_y+y+h), (0, 255, 0), 2)
    cv2.putText(output, str(digit), (roi_x+x, roi_y+y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# ----------------------------
# 8. Print results
# ----------------------------
print(f"Detected digits: {len(digit_contours)}")
print(f"Meter value: {compteur_final}")
if confidences:
    print(f"Average confidence: {np.mean(confidences):.2%}")

# ----------------------------
# 9. Visualization
# ----------------------------
fig, axes = plt.subplots(3, 4, figsize=(15, 10))

# Original image
axes[0,0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0,0].set_title("Original Image")
axes[0,0].axis('off')

# ROI grayscale
axes[0,1].imshow(roi, cmap='gray')
axes[0,1].set_title("ROI (Gray)")
axes[0,1].axis('off')

# ROI thresholded
axes[0,2].imshow(roi_thresh, cmap='gray')
axes[0,2].set_title("ROI Thresholded")
axes[0,2].axis('off')

# ROI cleaned
axes[0,3].imshow(roi_clean, cmap='gray')
axes[0,3].set_title("ROI Cleaned")
axes[0,3].axis('off')

# Output with rectangles
axes[1,0].imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
axes[1,0].set_title(f"Result: {compteur_final}")
axes[1,0].axis('off')

# Individual digits
for i, (x, y, w, h) in enumerate(digit_contours[:6]):
    digit_img = roi[y:y+h, x:x+w]
    _, digit_processed = cv2.threshold(digit_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    margin = 5
    digit_with_margin = np.zeros((h+2*margin, w+2*margin), dtype=np.uint8)
    digit_with_margin[margin:margin+h, margin:margin+w] = digit_processed
    if np.mean(digit_with_margin) < 127:
        digit_with_margin = cv2.bitwise_not(digit_with_margin)
    digit_rgb = cv2.cvtColor(digit_with_margin, cv2.COLOR_GRAY2RGB)
    digit_display = cv2.resize(digit_rgb, (40, 64))
    row = 1 + (i // 3)
    col = 1 + (i % 3)
    if row < 3:
        axes[row, col].imshow(digit_display)
        axes[row, col].set_title(f"Digit {i}: {compteur_final[i]}")
        axes[row, col].axis('off')

# Hide unused axes
for i in range(3):
    for j in range(4):
        if not axes[i, j].has_data():
            axes[i, j].axis('off')

plt.tight_layout()
plt.show()

# Final output alone
plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.title(f"Meter Detected: {compteur_final}")
plt.axis('off')
plt.show()
