import cv2
import mlflow
from ultralytics import YOLO
from utils import FPSCounter
from low_light import enhance_low_light

import cv2
cv2.setUseOptimized(True)
cv2.setNumThreads(4)


# -------------------------
# MLflow Setup
# -------------------------
mlflow.set_experiment("YOLOv8_RealTime_Object_Detection")
mlflow.start_run()

# -------------------------
# Load YOLOv8 pretrained model
# -------------------------
model = YOLO("runs/detect/train2/weights/best.onnx")

# -------------------------
# Webcam
# -------------------------
cap = cv2.VideoCapture(0)
fps_counter = FPSCounter()

print("🚀 Starting real-time object detection. Press 'Q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Low-light enhancement
    frame = enhance_low_light(frame)

    # Inference
    results = model(frame, conf=0.4)
    annotated_frame = results[0].plot()

    # FPS calculation
    fps = fps_counter.get_fps()

    # Log FPS to MLflow
    mlflow.log_metric("fps", fps)

    # Display FPS
    cv2.putText(
        annotated_frame,
        f"FPS: {int(fps)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("YOLOv8 Real-Time Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

mlflow.end_run()
print("🛑 Detection stopped.")
