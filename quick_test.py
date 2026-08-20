"""
quick_test.py
--------------
Runs a pretrained YOLOv8 model on one image, just to confirm everything
works before we do any custom training.
"""

from ultralytics import YOLO

# This downloads the pretrained model automatically the first time (small, ~6MB)
model = YOLO("yolov8n.pt")

# Run detection on any image - use a photo with people/objects in it
results = model("test.jpg")

# Save the image with boxes drawn on it
results[0].save("output.jpg")

# Print what it found
for box in results[0].boxes:
    class_name = model.names[int(box.cls)]
    confidence = float(box.conf)
    print(f"Detected: {class_name} ({confidence*100:.1f}% confident)")