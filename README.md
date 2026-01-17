# Water Meter Digit Detection using CNN (YOLOv8)

This project is a **Flask web application** that uses a **deep learning model (YOLOv8)** to detect and recognize **digits displayed on water meters** from images.
The system allows users to upload an image of a water meter and automatically extract the digit readings.

## 📌 Project Description

Manual reading of water meters is time-consuming and error-prone.
This project automates the process by using a **CNN-based object detection model (YOLOv8)** to detect meter digits in images and display the results through a **Flask web interface**.

## 🧠 Model

* **YOLOv8 Nano (`yolov8n.pt`)**
* CNN-based object detection architecture
* Trained / fine-tuned to detect water meter digits (0–9)

## 🛠 Technologies Used

* Python
* Flask
* YOLOv8 (Ultralytics)
* OpenCV
* NumPy

## 📂 Project Structure

```
├── app.py          # Flask web application
├── yolov8n.pt      # YOLOv8 pretrained or fine-tuned weights
├── requirements.txt
└── README.md
```

## 🚀 How to Run

### 1️⃣ Install dependencies

```bash
pip install flask ultralytics opencv-python numpy
```

### 2️⃣ Run the Flask app

```bash
python app.py
```

### 3️⃣ Open in browser

```
http://127.0.0.1:5000
```

Upload a water meter image and the detected digits will be displayed.

## 📊 Output

* Bounding boxes around detected digits
* Digit labels (0–9)
* Visualized results on the uploaded image

## 🔮 Future Improvements

* Improve accuracy with custom-trained datasets
* Add digit sequence reconstruction (full meter reading)
* Support real-time camera input
* Deploy using Docker or cloud services

## 📜 License

This project is intended for **academic and educational purposes**.
