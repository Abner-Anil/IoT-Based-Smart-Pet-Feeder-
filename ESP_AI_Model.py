from ultralytics import YOLO
import cv2
import serial
import time

# === Serial Setup ===
ser = serial.Serial('/dev/ttyUSB0', 115200)
time.sleep(2)  # give ESP time to reset

# === Load YOLOv8 model ===
model = YOLO("yolov8s.pt")

# === Open webcam ===
cap = cv2.VideoCapture(0)

last_detected = "None"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame, verbose=False)

    detected = "None"
    for box in results[0].boxes:
        cls = int(box.cls)
        label = model.names[cls]
        if label.lower() in ["cat", "dog"]:
            detected = label.capitalize()

    # Display detection on screen
    cv2.putText(frame, f"Detected: {detected}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Pet Detector", frame)

    # Print to terminal
    print("Detected:", detected)

    # === Send to ESP only if detection changed ===
    if detected != last_detected:
        msg = detected + "\n"
        ser.write(msg.encode('utf-8'))
        ser.flush()
        print("Sent to ESP:", detected)
        last_detected = detected

    # Exit if 'q' pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
ser.close()
