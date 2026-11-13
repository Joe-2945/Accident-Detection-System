# 🚗 AI-Based Accident Detection & Road Monitoring System

An AI-powered road monitoring application designed to detect road hazards and monitor accident-related information using YOLOv8, Python, and Flask.

## 📌 Project Overview

This project aims to improve road safety by combining computer vision and real-time road monitoring features.

The application uses a trained YOLOv8 model to identify road hazards and provides a web-based dashboard for monitoring detected events, vehicle location, and accident information.

## ✨ Key Features

- **AI-Based Detection:** YOLOv8-powered road hazard detection.
- **Image Detection:** Upload images for hazard analysis.
- **Live Video Monitoring:** Webcam-based video detection.
- **Sample Video Detection:** Process a sample video through the detection pipeline.
- **GPS Location Tracking:** Receive vehicle location updates through browser geolocation.
- **Hazard Monitoring:** Store detected hazards with coordinates, timestamps, and confidence values.
- **Accident Management:** Maintain and manage accident records.
- **Distance Calculation:** Calculate distances between vehicle location and reported hazards.
- **Dashboard Analytics:** Display detection statistics and training metrics.
- **Weather Information:** Retrieve weather information through Open-Meteo.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Backend programming |
| Flask | Web application framework |
| YOLOv8 | Object detection |
| OpenCV | Image and video processing |
| NumPy | Numerical operations |
| HTML, CSS, JavaScript | Frontend |
| JSON | Data storage |
| Open-Meteo API | Weather information |

## 📂 Project Structure

```text
Accident-Detection-System/
│
├── archive/              # Dataset and sample video files
├── runs/                 # Model training outputs and metrics
├── static/               # Static assets and uploads
├── templates/            # HTML templates
│
├── app.py                # Flask application
├── detect.py             # Hazard data utilities
├── train_model.py        # Model training script
│
├── best.pt               # Custom trained YOLO model
├── yolov8n.pt            # YOLOv8 base model
│
├── accidents.json        # Accident records
├── hazards.json          # Hazard records
├── requirements.txt      # Python dependencies
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Joe-2945/Accident-Detection-System.git
```

### 2. Navigate into the project

```bash
cd Accident-Detection-System
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

**Windows:**

```bash
venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the application

```bash
python app.py
```

Open your browser and navigate to:

```text
http://127.0.0.1:5000
```

Allow browser camera and location permissions when using the corresponding features.

## 🧠 Model Information

The project uses YOLOv8 for road hazard detection.

- `best.pt` – Custom trained model.
- `yolov8n.pt` – YOLOv8 nano base model.

The custom model is used when available, with the base model as a fallback.

## 🔮 Future Enhancements

- Real-time emergency alert integration.
- Automated emergency service notifications.
- Cloud-based data storage.
- Advanced accident severity classification.
- Improved model accuracy using larger datasets.
- Mobile application integration.
- Scalable deployment infrastructure.

## ⚠️ Disclaimer

This project is developed for educational and research purposes. Detection results should not be treated as a replacement for verified emergency reporting or professional road safety systems.

## 👨‍💻 Author

**Joe Pushparaj**

GitHub: https://github.com/Joe-2945

---

⭐ If you find this project interesting, consider giving the repository a star!
