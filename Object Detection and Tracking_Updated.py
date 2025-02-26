import os
# Fix potential library conflicts
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# Import YOLO from Ultralytics
from ultralytics import YOLO
# Load the YOLOv8 model (medium version)
model = YOLO('yolov8m.pt')
# Perform object tracking on the given video
# 'source' specifies the input video file
# 'show=True' displays the output in a window
# 'tracker="bytetrack.yaml"' enables the ByteTrack algorithm for tracking
results = model.track(source="Car_Video_Sample.mp4", show=True, tracker="bytetrack.yaml")
