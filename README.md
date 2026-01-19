# CNN-Based Water Meter Digit Recognition

This mini project uses a Convolutional Neural Network (CNN) to detect and recognize digits from water meter images. It also includes a Flask web app to upload images and display predictions. YOLOv8 can be optionally used for object detection and preprocessing of the meter region.

## **Team**
- Abdelaziz Ariri
- Ossama Ettaqafi

## **Project Structure**
```

CNN-Based-Water-Meter-Digit-Recognition/
│
├─ app.py              # Flask web application
├─ test.py             # Script to test the model on images
├─ yolov8n.pt          # YOLOv8 model (optional for meter detection)
├─ requirements.txt    # Project dependencies
├─ images/             # Example images
└─ README.md           # This file

````

## **Installation**

1. Clone the repository:

```bash
git clone https://github.com/ossama-ettaqafi/CNN-Based-Water-Meter-Digit-Recognition.git
cd CNN-Based-Water-Meter-Digit-Recognition
````

2. Create a virtual environment (Python 3.10 recommended):

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1   # Windows PowerShell
```

3. Upgrade pip:

```bash
python -m pip install --upgrade pip setuptools wheel
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

> ⚠️ **Note:** Do not install standalone Keras >= 3.x. TensorFlow 2.14 includes Keras compatible with Python 3.10.

## **Usage**

### **Run Flask app**

```bash
python app.py
```

1. Open your browser at `http://127.0.0.1:5000/`
2. Upload an image of a water meter
3. The app will display the detected digits

### **Test CNN model directly**

```bash
python test.py
```

* Loads an image from the `images/` folder
* Preprocesses it
* Predicts digits using `model_tf/`

## **Dependencies**

* Python 3.10
* TensorFlow 2.14 (includes Keras 2.14.x)
* NumPy
* Matplotlib
* Pillow
* OpenCV
* Flask >= 3.1.2
* PyTorch >= 2.2.0
* Torchvision >= 0.17.1
* Torchaudio >= 2.2.0
* Ultralytics (YOLOv8) >= 8.0.150

## **Notes**

* Python 3.10 is recommended for TensorFlow compatibility.
* YOLOv8 is optional but helps improve meter digit localization.
* Make sure your virtual environment is activated before running scripts.
* Your `model.keras` should be converted to **TensorFlow format** (`model_tf/`) for Python 3.10 compatibility.

## **License**

This project is for educational and research purposes.