import cv2
import torch
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLO model (YOLOv8)
model = YOLO('yolov8n.pt')  # Use yolov8s.pt, yolov8m.pt, etc. for different sizes

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0)

# Open video capture (0 for webcam, or provide video file path)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run YOLO detection
    results = model(frame)
    detections = []
    
    for result in results:
        for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
            x1, y1, x2, y2 = map(int, box)
            width, height = x2 - x1, y2 - y1  # Convert to correct format
            confidence = conf.item()
            class_id = int(cls.item())  # Convert class to integer
            class_name = model.names[class_id]  # Get class label
            
            # Append in DeepSORT format ([left, top, width, height], confidence, class_name)
            detections.append(([x1, y1, width, height], confidence, class_name))
    
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
    
    # Display frame
    cv2.imshow('Object Detection & Tracking', frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
