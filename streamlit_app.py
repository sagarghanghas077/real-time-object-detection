import streamlit as st
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO
from src.low_light import enhance_low_light

# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="Real-Time Object Detection",
    layout="wide"
)

st.title("🚗 Real-Time Object Detection using YOLOv8")
st.write("Upload an image or video to detect objects like vehicles and people.")

# -------------------------
# Load Model
# -------------------------
@st.cache_resource
def load_model():
    # Use ONNX model if available, else fallback
    try:
        return YOLO("runs/detect/train2/weights/best.onnx")
    except:
        return YOLO("yolov8n.pt")

model = load_model()

# -------------------------
# Sidebar
# -------------------------
st.sidebar.header("Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.4)

upload_type = st.sidebar.radio(
    "Choose input type",
    ["Image", "Video"]
)

# -------------------------
# IMAGE INFERENCE
# -------------------------
if upload_type == "Image":
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        image = enhance_low_light(image)

        results = model(image, conf=conf_threshold)
        annotated = results[0].plot()

        st.image(annotated, channels="BGR", caption="Detected Objects")

# -------------------------
# VIDEO INFERENCE
# -------------------------
else:
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = enhance_low_light(frame)

            results = model(frame, conf=conf_threshold)
            annotated = results[0].plot()

            stframe.image(annotated, channels="BGR")

        cap.release()
