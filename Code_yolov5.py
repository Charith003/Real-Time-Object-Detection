import cv2
from ultralytics import YOLO

# Load the YOLO model (you can choose a YOLOv5 version like "yolov5s.pt")
model = YOLO("yolov5s.pt")  # You can use "yolov5n.pt", "yolov5m.pt", or "yolov5l.pt"

# Start webcam
webcamera = cv2.VideoCapture(0)

while True:
    success, frame = webcamera.read()  # Read a frame from the webcam
    if not success:
        print("Failed to capture image")
        break

    # Perform object detection
    results = model(frame)  # Run detection on the frame

    # Get the frame with detected bounding boxes and labels
    frame_with_boxes = results[0].plot()  # Plot bounding boxes on the frame

    # Display the resulting frame
    cv2.imshow("YOLO Object Detection", frame_with_boxes)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the webcam and close all OpenCV windows
webcamera.release()
cv2.destroyAllWindows()

