import cv2
import torch
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLO model (YOLOv8)
model = YOLO('yolov8n.pt')  # You can use yolov8s.pt, yolov8m.pt, etc.

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0)

# Open video capture
cap = cv2.VideoCapture(0)  # Use 0 for webcam or provide a video file path

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run YOLO detection
    results = model(frame)
    detections = []
    for result in results:
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            detections.append(([x1, y1, x2, y2], 1.0, 'object'))
    
    # Run DeepSORT tracking
    tracks = tracker.update_tracks(detections, frame=frame)
    
    for track in tracks:
        if track.is_confirmed():
            bbox = track.to_tlbr()
            x1, y1, x2, y2 = map(int, bbox)
            track_id = track.track_id
            
            # Draw bounding box and ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imshow('Object Detection & Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
