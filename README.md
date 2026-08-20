# Pothole Detection with YOLOv8

An object detection system that identifies potholes in road images, built end-to-end: 
custom dataset, fine-tuned YOLOv8 model, and a FastAPI service for real-time inference.

## What it does

Takes an image of a road and returns bounding boxes + confidence scores for every 
pothole detected.

## Tech stack

- **Model:** YOLOv8 (Ultralytics), fine-tuned on a custom single-class dataset
- **Preprocessing:** OpenCV
- **API:** FastAPI
- **Training:** Google Colab (T4 GPU)
- **Dataset:** 1,482 annotated images (Roboflow Universe - Pothole Detection)

## Results

| Metric      | Score |
|-------------|-------|
| Precision   | 0.87  |
| Recall      | 0.82  |
| mAP50       | 0.90  |
| mAP50-95    | 0.57  |

## How to run it

1. Install dependencies: