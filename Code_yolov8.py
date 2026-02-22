import cv2
from ultralytics import YOLO

# Load a pretrained YOLOv8n model (nano = fast and lightweight)
model = YOLO("yolov8n.pt")  # You can also try 'yolov8s.pt'

# Open webcam (0 = default webcam)
cap = cv2.VideoCapture(0)

# Check if the webcam opens correctly
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Run YOLOv8 detection on the frame
    results = model(frame, show=False)  # You can set show=True to auto display

    # Visualize detections on the frame
    annotated_frame = results[0].plot()

    # Show frame with detections
    cv2.imshow("YOLOv8 Detection", annotated_frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release everything
cap.release()
cv2.destroyAllWindows()
