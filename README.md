
## **🛠 Object Detection and Tracking**
### **🔍 Real-Time Object Detection & Tracking using YOLO / Faster R-CNN**

![Object Detection](https://miro.medium.com/max/1400/1*VzVVZKkevcvUNX70C0YB2g.gif)  

---

### **📖 Overview**
This project implements **real-time object detection and tracking** using deep learning models like **YOLO (You Only Look Once)** and **Faster R-CNN**. It processes video streams and identifies objects with high accuracy.  

---

### **⚡ Features**
✅ Real-time object detection  
✅ Supports YOLOv8, Faster R-CNN  
✅ Tracks multiple objects simultaneously  
✅ Custom model training support  
✅ Works with webcam or video files  

---

### **🔧 Installation**
#### **1️⃣ Install Dependencies**
Run the following command:  
```bash
pip install -r requirements.txt
```

#### **2️⃣ Clone the Repository**
```bash
git clone https://github.com/lillibot2054/Object-Detection-and-Tracking.git
cd Object-Detection-and-Tracking
```

#### **3️⃣ Run the Object Detection Script**
```bash
python Object-Detection-and-Tracking.py
```
### 🔹 Updated Version: `Object Detection and Tracking_Updated.py`  
- **Improved accuracy and faster detection**  
- **Better object tracking with optimized performance**  
- **Fine-tuned confidence thresholds to reduce false positives**  

---

### **📌 Requirements**
- Python 3.8+
- OpenCV
- PyTorch or TensorFlow
- YOLOv8 / Faster R-CNN
- NumPy
- Matplotlib

---

### **🎯 Usage**
- **Run on Webcam:**  
  ```bash
  python object_tracking.py --source 0
  ```
- **Run on Video File:**  
  ```bash
  python object_tracking.py --source videos/sample.mp4
  ```
- **Train a Custom Model:**  
  ```bash
  python train.py --data custom.yaml --epochs 50
  ```
### 🔄 **Update: Object Detection and Tracking Improvements**  

#### **Enhancements in `Object Detection and Tracking.py`**  
- **Refined Bounding Box Format**: Adjusted bounding box coordinates to match DeepSORT's expected input format `[left, top, width, height]`.  
- **Improved YOLO Model Integration**: Enhanced detection reliability by correctly parsing and utilizing YOLO outputs.  
- **Better Tracking Performance**: Optimized DeepSORT parameters for smoother and more accurate object tracking.  
- **Bug Fixes**: Resolved issues where incorrect bounding box dimensions could affect tracking consistency.  

These updates improve the overall stability, accuracy, and efficiency of the object detection and tracking system. 🚀
---

### **Contributors**
- **DHEEPAK.G** - Developer  
- **GitHub:** [@lillibot2054](https://github.com/lillibot2054)  

---

### **📜 License**
This project is licensed under the **MIT License** – feel free to modify and use!  

---

### **⭐ Star This Repo**
If you like this project, don't forget to ⭐ star the repository! 😊✨  
