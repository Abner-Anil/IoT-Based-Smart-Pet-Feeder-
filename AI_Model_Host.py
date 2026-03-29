from ultralytics import YOLO
import cv2
import serial
import time
import flask
from flask import Flask, Response

# ==========================
# SERIAL SETUP
# ==========================
ser = serial.Serial('/dev/ttyUSB0', 115200)
time.sleep(2)

# ==========================
# LOAD YOLO MODEL
# ==========================
model = YOLO("yolov8s.pt")

# ==========================
# CAMERA SETUP
# ==========================
cap = cv2.VideoCapture(0)

last_detected = "None"

# ==========================
# FLASK APP
# ==========================
app = Flask(__name__)


def generate_frames():
    global last_detected

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # === YOLO Detection ===
        results = model(frame, verbose=False)

        detected = "None"
        for box in results[0].boxes:
            cls = int(box.cls)
            label = model.names[cls]
            if label.lower() in ["cat", "dog"]:
                detected = label.capitalize()

        # === Draw Result on Frame ===
        cv2.putText(frame, f"Detected: {detected}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # === Send update to ESP32 ONLY if changed ===
        if detected != last_detected:
            msg = detected + "\n"
            ser.write(msg.encode('utf-8'))
            ser.flush()
            print("Sent to ESP:", detected)
            last_detected = detected

        # === Encode frame as JPEG for streaming ===
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # === MJPEG Stream ===
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    return "Pet Detector Streaming Server Running!"


if __name__ == '__main__':
    # Make visible to entire WiFi network
    app.run(host='0.0.0.0', port=5000)
