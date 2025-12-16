# Real-Time Object Detection System

## Overview
A real-time object detection system built using YOLOv8 to detect vehicles,
pedestrians, and common objects from live video streams.

## Tech Stack
- YOLOv8 (Ultralytics)
- Python
- OpenCV
- ONNX Runtime
- MLflow

## Features
- Real-time webcam inference
- Low-light image enhancement
- FPS and latency monitoring
- Training on coco128 dataset
- Optimized inference using ONNX Runtime

## Installation
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
