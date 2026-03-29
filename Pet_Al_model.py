from ultralytics import YOLO
import cv2

# Load YOLOv8 small pre-trained model
model = YOLO("yolov8s.pt")

cap = cv2.VideoCapture(0)  # open laptop camera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame, verbose=False)

    detected = "None"
    for box in results[0].boxes:
        cls = int(box.cls)
        label = model.names[cls]
        if label.lower() in ["cat", "dog"]:
            detected = label.capitalize()

    # Show on screen
    cv2.putText(frame, f"Detected: {detected}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Pet Detector", frame)

    print("Detected:", detected)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
