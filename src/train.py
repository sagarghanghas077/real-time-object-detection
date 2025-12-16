
from ultralytics import YOLO

# Load pretrained YOLOv8 model
model = YOLO("yolov8n.pt")

# Train on coco128 (CPU)
results = model.train(
    data="coco128.yaml",
    epochs=20,
    imgsz=640,
    batch=16,
    device="cpu"
)

print("✅ Training completed successfully")
