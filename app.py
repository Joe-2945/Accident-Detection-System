"""
app.py - Smart AI-Based Road Monitoring System (Full Pro Version)

Routes:
  GET  /                     → Dashboard
  POST /upload               → Image detection
  GET  /video_feed           → Live webcam MJPEG
  GET  /video_demo           → sample_video.mp4 MJPEG
  POST /api/location         → Update vehicle GPS from browser
  GET  /api/hazards          → Hazard list (with distances)
  GET  /api/stats            → Dashboard counters
  GET  /api/accidents        → List active accidents
  POST /api/accidents        → Add accident (expires in 24h)
  DEL  /api/accidents/<id>   → Remove accident
  GET  /api/traffic          → Simulated traffic density zones
  GET  /api/metrics          → YOLOv8 training results from CSV
  GET  /api/weather          → Proxied Open-Meteo weather data
"""

import os, cv2, json, math, time, uuid, threading, csv
import numpy as np
import requests
from flask import Flask, render_template, Response, request, jsonify
from ultralytics import YOLO
from datetime import datetime, timedelta
from detect import save_hazard, HAZARDS_FILE

app = Flask(__name__)
app.jinja_env.variable_start_string = '[['
app.jinja_env.variable_end_string = ']]'

# ==============================
# PATHS & CONFIG
# ==============================
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER  = os.path.join(BASE_DIR, 'static', 'uploads')
SAMPLE_VIDEO   = os.path.join(BASE_DIR, 'archive', 'sample_video.mp4')
ACCIDENTS_FILE = os.path.join(BASE_DIR, 'accidents.json')
RESULTS_CSV    = os.path.join(BASE_DIR, 'runs', 'detect', 'pothole_detector', 'results.csv')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Hazard throttle: save at most once per N seconds during streaming
SAVE_INTERVAL = 3
_last_save = {"webcam": 0, "demo": 0}

# Thread locks
_json_lock    = threading.Lock()
_vehicle_lock = threading.Lock()

# ==============================
# VEHICLE STATE (thread-safe)
# Updated live via POST /api/location from browser GPS
# ==============================
_vehicle = {
    "lat":        10.7905,   # Default: Tiruchirappalli
    "lon":        78.7047,
    "updated_at": None       # None = using default, not live GPS
}

def get_vehicle():
    with _vehicle_lock:
        return _vehicle["lat"], _vehicle["lon"]


# ==============================
# LOAD YOLO MODEL
# ==============================
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")
if not os.path.exists(MODEL_PATH):
    print("⚠️ best.pt not found — falling back to yolov8n.pt. Run train_model.py first.")
    MODEL_PATH = "yolov8n.pt"
else:
    print(f"✅ Loaded model: best.pt")

model = YOLO(MODEL_PATH)


# ==============================
# UTILITY FUNCTIONS
# ==============================

def haversine(lat1, lon1, lat2, lon2):
    """Haversine distance in metres."""
    R = 6_371_000
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    Δφ = math.radians(lat2 - lat1)
    Δλ = math.radians(lon2 - lon1)
    a = math.sin(Δφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(Δλ/2)**2
    return round(2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a)), 1)


def load_hazards():
    """Load all hazards from JSON file safely."""
    if not os.path.exists(HAZARDS_FILE):
        return []
    with _json_lock:
        try:
            with open(HAZARDS_FILE) as f:
                return json.load(f)
        except:
            return []


def load_accidents():
    """
    Load accidents, automatically discarding entries that have expired
    (older than 24 hours from creation).
    """
    if not os.path.exists(ACCIDENTS_FILE):
        return []
    try:
        with open(ACCIDENTS_FILE) as f:
            data = json.load(f)
    except:
        return []
    now = datetime.now()
    valid = [a for a in data
             if datetime.strptime(a["expires_at"], "%Y-%m-%d %H:%M:%S") > now]
    # If some entries expired, persist the cleaned list
    if len(valid) != len(data):
        _save_accidents(valid)
    return valid


def _save_accidents(data):
    with open(ACCIDENTS_FILE, "w") as f:
        json.dump(data, f, indent=2)


def run_yolo(frame, draw=True):
    """
    Run YOLOv8 inference on a BGR frame.
    Uses imgsz=320 for much faster CPU inference.
    Returns (annotated_frame, list_of_detections).
    """
    results = model(frame, conf=0.40, verbose=False, imgsz=320)[0]
    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf  = float(box.conf[0])
        label = model.names[int(box.cls[0])]
        if draw:
            cv2.rectangle(frame, (x1, y2 - 26), (x2, y2), (220, 38, 38), -1)
            cv2.rectangle(frame, (x1, int(y1)), (x2, y2), (220, 38, 38), 2)
            cv2.putText(frame, f"{label}  {conf:.0%}", (x1 + 4, y2 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        detections.append({
            "type": label, 
            "confidence": round(conf, 3), 
            "box": (x1, int(y1), x2, y2)
        })
    return frame, detections


def maybe_save_hazard(detections, key):
    """Save one hazard per SAVE_INTERVAL seconds to avoid flooding the JSON."""
    global _last_save
    if not detections:
        return
    if time.time() - _last_save[key] < SAVE_INTERVAL:
        return
    d = detections[0]
    vlat, vlon = get_vehicle()
    lat = vlat + np.random.uniform(-0.0008, 0.0008)
    lon = vlon + np.random.uniform(-0.0008, 0.0008)
    with _json_lock:
        save_hazard(d["type"], lat, lon, d["confidence"])
    _last_save[key] = time.time()


def encode_frame(frame):
    """Encode BGR frame as JPEG bytes for MJPEG streaming."""
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
    return buf.tobytes()


# ======================================================================
# CAMERA PIPELINE  (two‑thread architecture)
#
#  Thread 1  (_capture_thread)  — reads raw frames from the camera as
#      fast as the device allows and writes them into a shared buffer.
#      The first frame appears in < 100 ms so the user sees the camera
#      immediately after clicking "Start Camera".
#
#  Thread 2  (_yolo_thread)  — picks up the latest raw frame, runs
#      inference (slow), then writes detection overlays into a second
#      shared buffer.  This thread is completely decoupled from the
#      capture thread and never blocks it.
#
#  stream_generator  — the MJPEG HTTP generator reads whichever buffer
#      has the freshest data (annotated if available, raw otherwise)
#      and yields it at a capped ~15 FPS.
# ======================================================================

STREAM_W = 480         # Resize all frames to this width for speed

# Per‑stream shared state
_streams = {}          # key → dict (see _init_stream)


def _init_stream(key):
    """Create a fresh shared‑state dict for a stream key."""
    _streams[key] = {
        "raw_frame":     None,  # latest raw BGR numpy array
        "encoded_bytes": None,  # latest JPEG bytes (with overlay drawn)
        "latest_dets":   [],    # freshest bounding boxes from YOLO
        "lock":          threading.Lock(),
        "active":        True,
    }


def _capture_thread(source, key):
    """
    Reads frames as fast as the device delivers them.
    Instantly overlays the last known YOLO boxes onto the live feed.
    """
    # OpenCV Windows defaults to MSMF which has huge startup delays + latency
    if source == 0 and os.name == 'nt':
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(source)
        
    if not cap.isOpened():
        ph = np.zeros((360, 640, 3), dtype=np.uint8)
        msg = "No webcam found" if source == 0 else "Source unavailable"
        cv2.putText(ph, msg, (120, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (70, 70, 70), 2)
        with _streams[key]["lock"]:
            _streams[key]["encoded_bytes"] = encode_frame(ph)
        return

    # Lower webcam resolution for speed and prevent buffering lag
    if source == 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)    # minimize internal buffer

    while _streams[key]["active"]:
        ok, frame = cap.read()
        if not ok:
            if isinstance(source, str):         # loop video files
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break

        # Resize to target width
        h, w = frame.shape[:2]
        if w > STREAM_W:
            scale = STREAM_W / w
            frame = cv2.resize(frame, (STREAM_W, int(h * scale)),
                               interpolation=cv2.INTER_NEAREST)

        # Get latest detections lock-free (fast)
        with _streams[key]["lock"]:
            dets = _streams[key]["latest_dets"]
            _streams[key]["raw_frame"] = frame.copy() # Provide clean frame to YOLO

        # Draw asynchronous boxes on top of the live feed
        display_frame = frame
        for d in dets:
            x1, y1, x2, y2 = d["box"]
            cv2.rectangle(display_frame, (x1, y2 - 26), (x2, y2), (220, 38, 38), -1)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (220, 38, 38), 2)
            cv2.putText(display_frame, f'{d["type"]}  {d["confidence"]:.0%}', (x1 + 4, y2 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        # Encode *outside* the lock so we don't stall the HTTP generator
        encoded_jpg = encode_frame(display_frame)

        with _streams[key]["lock"]:
            _streams[key]["encoded_bytes"] = encoded_jpg

    cap.release()


def _yolo_thread(key):
    """
    Picks up the latest clean raw frame asynchronously, runs YOLO, and writes
    only the new bounding boxes back to the shared state.
    """
    last_id = None                      # id() of the last processed frame
    while _streams[key]["active"]:
        with _streams[key]["lock"]:
            raw = _streams[key]["raw_frame"]

        if raw is None or id(raw) == last_id:
            time.sleep(0.01)            # nothing new yet
            continue

        last_id = id(raw)
        frame = raw.copy()              # snapshot so capture can keep writing
        
        # Don't natively draw boxes to save CPU time on YOLO thread
        _, dets = run_yolo(frame, draw=False)
        maybe_save_hazard(dets, key)

        with _streams[key]["lock"]:
            _streams[key]["latest_dets"] = dets


def stream_generator(source, key):
    """
    MJPEG HTTP generator.
    • Reads the pre-encoded raw frame (which includes instant overlay).
    • Outputs at 30 FPS cap for extreme buttery smoothness.
    """
    # ---- start / restart threads ----
    if key in _streams and _streams[key]["active"]:
        _streams[key]["active"] = False
        time.sleep(0.15)                # let old threads exit

    _init_stream(key)
    threading.Thread(target=_capture_thread, args=(source, key), daemon=True).start()
    threading.Thread(target=_yolo_thread, args=(key,), daemon=True).start()

    try:
        while _streams[key]["active"]:
            with _streams[key]["lock"]:
                encoded_bytes = _streams[key]["encoded_bytes"]
                
            if encoded_bytes:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                       + encoded_bytes + b"\r\n")
            time.sleep(1/30.0)          # Max 30 FPS output
    finally:
        _streams[key]["active"] = False


def get_traffic_zones():
    """
    Simulate traffic density based on time of day.
    Peak hours (7-10 AM, 5-8 PM) → high density.
    Midday → medium.  Night → low.
    """
    hour = datetime.now().hour
    vlat, vlon = get_vehicle()

    # 5 named zones around the vehicle
    zone_offsets = [
        (0.003,  0.002,  "Main Junction"),
        (-0.002, 0.004,  "Market Area"),
        (0.005,  -0.003, "Highway Merge"),
        (-0.004, -0.005, "School Zone"),
        (0.001,  0.006,  "Bus Terminal"),
    ]
    zones = []
    for dlat, dlon, name in zone_offsets:
        if (7 <= hour <= 10) or (17 <= hour <= 20):
            level = np.random.choice(["high", "medium"], p=[0.7, 0.3])
        elif 11 <= hour <= 16:
            level = np.random.choice(["medium", "low"], p=[0.6, 0.4])
        else:
            level = "low"

        color = {"high": "#dc2626", "medium": "#f97316", "low": "#22c55e"}[level]
        vehicles = {"high": int(np.random.randint(80, 150)),
                    "medium": int(np.random.randint(30, 79)),
                    "low": int(np.random.randint(5, 29))}[level]
        zones.append({
            "lat": round(vlat + dlat, 6),
            "lon": round(vlon + dlon, 6),
            "level": level, "color": color,
            "name": name, "vehicles": vehicles
        })
    return zones


def read_metrics_csv():
    """
    Parse YOLO training results.csv.
    Strips whitespace from headers (Ultralytics pads them).
    Returns list of dicts, or None if file not found.
    """
    if not os.path.exists(RESULTS_CSV):
        return None
    rows = []
    with open(RESULTS_CSV, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k.strip(): v.strip() for k, v in row.items()})
    return rows or None


# ==============================
# FLASK ROUTES
# ==============================

@app.route("/")
def index():
    vlat, vlon = get_vehicle()
    hazards = load_hazards()
    for h in hazards:
        h["distance"] = haversine(vlat, vlon, h["latitude"], h["longitude"])
    alerts = [f"⚠️ {h['hazard_type'].capitalize()} detected {h['distance']} m ahead"
              for h in hazards if h["distance"] <= 100]
    return render_template("index.html",
                           hazards=hazards, alerts=alerts,
                           v_lat=vlat, v_lon=vlon)


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files or not request.files["file"].filename:
        return "No file provided", 400
    file = request.files["file"]
    orig_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(orig_path)

    img = cv2.imread(orig_path)
    processed, dets = run_yolo(img)
    out_name = "det_" + file.filename
    cv2.imwrite(os.path.join(UPLOAD_FOLDER, out_name), processed)

    vlat, vlon = get_vehicle()
    for d in dets:
        lat = vlat + np.random.uniform(-0.0008, 0.0008)
        lon = vlon + np.random.uniform(-0.0008, 0.0008)
        with _json_lock:
            save_hazard(d["type"], lat, lon, d["confidence"])

    hazards = load_hazards()
    for h in hazards:
        h["distance"] = haversine(vlat, vlon, h["latitude"], h["longitude"])
    alerts = [f"⚠️ {h['hazard_type'].capitalize()} {h['distance']} m ahead"
              for h in hazards if h["distance"] <= 100]
    return render_template("index.html",
                           hazards=hazards, alerts=alerts,
                           uploaded_img=out_name, detection_count=len(dets),
                           v_lat=vlat, v_lon=vlon)


@app.route("/video_feed")
def video_feed():
    return Response(stream_generator(0, "webcam"),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/video_demo")
def video_demo():
    return Response(stream_generator(SAMPLE_VIDEO, "demo"),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# ---- API: Location ----
@app.route("/api/location", methods=["POST"])
def update_location():
    """Receive GPS coordinates from the browser's Geolocation API."""
    data = request.get_json(silent=True) or {}
    if "lat" not in data or "lon" not in data:
        return jsonify({"error": "lat and lon required"}), 400
    with _vehicle_lock:
        _vehicle["lat"] = float(data["lat"])
        _vehicle["lon"] = float(data["lon"])
        _vehicle["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return jsonify({"status": "ok", "lat": _vehicle["lat"], "lon": _vehicle["lon"]})


# ---- API: Hazards ----
@app.route("/api/hazards")
def api_hazards():
    vlat, vlon = get_vehicle()
    hazards = load_hazards()
    for h in hazards:
        h["distance"] = haversine(vlat, vlon, h["latitude"], h["longitude"])
    return jsonify(hazards)


# ---- API: Stats ----
@app.route("/api/stats")
def api_stats():
    vlat, vlon = get_vehicle()
    hazards = load_hazards()
    nearby = sum(1 for h in hazards
                 if haversine(vlat, vlon, h["latitude"], h["longitude"]) <= 100)
    avg_conf = (round(sum(h.get("confidence", 0) for h in hazards)
                     / len(hazards) * 100) if hazards else 0)
    return jsonify({
        "total":        len(hazards),
        "nearby":       nearby,
        "avg_conf":     avg_conf,
        "v_lat":        vlat,
        "v_lon":        vlon,
        "location_live": _vehicle["updated_at"] is not None
    })


# ---- API: Accidents ----
@app.route("/api/accidents", methods=["GET"])
def get_accidents():
    return jsonify(load_accidents())


@app.route("/api/accidents", methods=["POST"])
def add_accident():
    data = request.get_json(silent=True) or {}
    vlat, vlon = get_vehicle()
    accident = {
        "id":          str(uuid.uuid4()),
        "lat":         float(data.get("lat", vlat)),
        "lon":         float(data.get("lon", vlon)),
        "description": str(data.get("description", "Accident reported"))[:200],
        "timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "expires_at":  (datetime.now() + timedelta(hours=24))
                       .strftime("%Y-%m-%d %H:%M:%S")
    }
    accidents = load_accidents()
    accidents.append(accident)
    _save_accidents(accidents)
    return jsonify(accident), 201


@app.route("/api/accidents/<aid>", methods=["DELETE"])
def delete_accident(aid):
    accidents = [a for a in load_accidents() if a["id"] != aid]
    _save_accidents(accidents)
    return jsonify({"status": "deleted"})


# ---- API: Traffic ----
@app.route("/api/traffic")
def api_traffic():
    return jsonify(get_traffic_zones())


# ---- API: Model Metrics ----
@app.route("/api/metrics")
def api_metrics():
    rows = read_metrics_csv()
    if not rows:
        return jsonify({"available": False})
    return jsonify({"available": True, "rows": rows})


# ---- API: Weather (via Open-Meteo, free, no API key) ----
_weather_cache = {"data": None, "ts": 0}   # simple in-memory cache
WEATHER_CACHE_TTL = 300                     # 5 minutes

@app.route("/api/weather")
def api_weather():
    vlat, vlon = get_vehicle()

    # Return cached result if fresh enough
    if _weather_cache["data"] and (time.time() - _weather_cache["ts"] < WEATHER_CACHE_TTL):
        return jsonify(_weather_cache["data"])

    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={vlat}&longitude={vlon}"
            f"&current=temperature_2m,apparent_temperature,"
            f"relative_humidity_2m,wind_speed_10m,weather_code,"
            f"precipitation,visibility"
            f"&timezone=auto"
        )
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        result = resp.json()
        _weather_cache["data"] = result
        _weather_cache["ts"] = time.time()
        return jsonify(result)
    except Exception as e:
        # Return cached data if available, otherwise offline fallback
        if _weather_cache["data"]:
            return jsonify(_weather_cache["data"])
        # Offline fallback with realistic placeholder data
        return jsonify({
            "current": {
                "temperature_2m": 28,
                "apparent_temperature": 30,
                "relative_humidity_2m": 65,
                "wind_speed_10m": 8,
                "weather_code": 0,
                "precipitation": 0,
                "visibility": 10000
            },
            "_offline": True,
            "_note": "Offline mode — showing estimated data"
        })


if __name__ == "__main__":
    print("\n🚀  Smart Road Monitoring System")
    print("📍  http://127.0.0.1:5000\n")
    app.run(debug=True, threaded=True)
