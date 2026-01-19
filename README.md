# CNN-Based Water Meter Digit Recognition

This project uses a Convolutional Neural Network (CNN) to detect and recognize digits from water meter images. It also includes a Flask web app to upload images and display predictions. YOLOv8 can be used for object detection and preprocessing of the meter region.

## **Team**

- Abdelaziz Ariri  
- Ossama Ettaqafi  

## **Project Structure**

```

CNN-Based-Water-Meter-Digit-Recognition/
│
├─ app.py              # Flask web application
├─ test.py             # Script to test the model on images
├─ model.keras         # Trained CNN model
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

## **Usage**

### **Run Flask app**

```bash
python app.py
```

* Open your browser at `http://127.0.0.1:5000/`
* Upload an image of a water meter
* The app will display the detected digits

### **Test CNN model directly**

```bash
python test.py
```

* Loads an image from the `images/` folder
* Preprocesses it
* Predicts digits using `model.keras`

## **Dependencies**

* TensorFlow 2.14
* Keras >=3.0
* NumPy
* Matplotlib
* Pillow
* OpenCV
* Flask
* PyTorch
* Ultralytics (YOLOv8)

## **Notes**

* Python 3.10 is recommended for TensorFlow compatibility.
* YOLOv8 is optional but helps improve meter digit localization.
* Make sure your venv is activated before running scripts.

## **License**

This project is for educational and research purposes.