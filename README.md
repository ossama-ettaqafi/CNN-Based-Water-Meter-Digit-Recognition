# Precise Water Meter OCR

![⚠ Experimental](https://img.shields.io/badge/Status-Experimental-orange)
![🛠 Work in Progress](https://img.shields.io/badge/Development-WIP-blue)

A **Flask-based web application** for precise reading extraction from water meter images using **OpenCV** and **Tesseract OCR**.
The application focuses on **high-accuracy digit detection**, offering per-digit confidence and annotated visual output.

---

## 🔹 Features

* Targeted **Region of Interest (ROI)** for precise meter digit extraction
* **Contrast enhancement** and **adaptive thresholding** for better digit isolation
* **Contour-based digit detection** with shape and size filtering
* **Individual digit OCR** for maximum accuracy
* **Confidence scoring** for each digit and overall reading
* Annotated output image showing:

  * Detected digits
  * Bounding boxes
  * Overall reading
  * Confidence
* Debug images saved for intermediate steps
* JSON API response with detailed digit-level information

---

## 🔹 App Screenshot

Here’s an example of the app in action:

<img width="600" alt="App Screenshot" src="https://github.com/user-attachments/assets/d5e9bf85-59e9-4393-b9f7-36128abc3284" />

**Description:**

* **Upload Section:** Upload water meter images.
* **Result Display:** Annotated meter image with detected digits and reading.
* **Digit Details:** Per-digit confidence and bounding boxes (optional, shown in debug images).
* **JSON Output:** API returns meter reading, confidence, and annotated image URL.

---

## 🔹 Requirements

* Python 3.10+
* Flask
* OpenCV (`cv2`)
* pytesseract
* numpy
* Pillow (optional for image handling)

### Install dependencies

```bash
pip install flask opencv-python pytesseract numpy
```

**Tesseract OCR** must be installed:
[Download Tesseract](https://github.com/tesseract-ocr/tesseract)

Set the path in the app:

```python
pytesseract.pytesseract.tesseract_cmd = r"D:\Program Files\Tesseract-OCR\tesseract.exe"
```

---

## 🔹 Project Structure

```
.
├── app.py                # Main Flask application
├── uploads/              # Uploaded and result images
├── debug/                # Debug intermediate images
├── templates/
│   └── index.html        # Frontend template
├── static/               # Optional CSS/JS files
└── README.md             # This file
```

---

## 🔹 Usage

### Start the server

```bash
python app.py
```

Visit: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

### API Endpoints

| Route                 | Method | Description                        |
| --------------------- | ------ | ---------------------------------- |
| `/`                   | GET    | Frontend page for uploading images |
| `/upload`             | POST   | Upload image for processing        |
| `/uploads/<filename>` | GET    | Serve processed result image       |
| `/debug/<filename>`   | GET    | Serve intermediate debug image     |
| `/debug_images`       | GET    | List last 10 debug images          |
| `/health`             | GET    | Health check                       |

---

### Example Upload Request (cURL)

```bash
curl -X POST -F "image=@meter.jpg" http://127.0.0.1:5000/upload
```

**Sample JSON Response**

```json
{
  "success": true,
  "reading": "00012345",
  "confidence": "88.5%",
  "image_url": "/uploads/result_20260127_170102.jpg",
  "debug_url": "/debug/20260127_170102_original_roi.jpg",
  "digit_count": 8,
  "digit_details": [
    {"position":1,"digit":"0","confidence":"92.0%","detected":true,"status":"✓"},
    ...
  ],
  "timestamp": "2026-01-27 17:01:02",
  "debug_timestamp": "20260127_170102"
}
```

---

## 🔹 How It Works (Pipeline)

**High-level pipeline:**

```
Upload → ROI → Enhance → Segment → OCR → Assemble Reading → Visualize → Output
```

**Step Description:**

1. **Upload:** User uploads water meter image
2. **ROI:** Extract the region containing digits
3. **Enhance:** Improve contrast for clearer digit detection
4. **Segment:** Apply thresholding and contour detection to isolate digits
5. **OCR:** Recognize each digit individually
6. **Assemble Reading:** Combine digits into the final reading
7. **Visualize:** Annotate digits and reading on the image
8. **Output:** Return JSON response with reading, confidence, and image URLs

---

## 🔹 Debugging & Development

* Debug images stored in `debug/` folder:

  * Original ROI
  * Binary threshold image
  * Individual digit crops
* Access last 10 debug images: `/debug_images`

---

## 🔹 Notes

* Optimized for **mechanical water meters**
* Works best with **high-contrast black digits on white background**
* Maximum upload size: **16 MB**
* Confidence scoring helps flag uncertain readings for manual review

---

## 🔹 License

MIT License – free for academic and commercial use
