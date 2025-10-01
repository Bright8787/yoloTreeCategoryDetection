import cv2
from ultralytics import YOLO
from pathlib import Path
# Load YOLO11n pretrained detection model
model = YOLO("yolo11n.pt")
cap = cv2.VideoCapture(0)  # 0 = default camera
# Run real-time detection on webcam

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model.predict(source=0, show=True, conf=0.5)
    annotated_frame = results[0].plot()
    cv2.imshow("YOLO Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()