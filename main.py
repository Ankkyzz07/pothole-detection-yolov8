from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import numpy as np
import cv2
from ultralytics import YOLO

app = FastAPI()
model = YOLO("best.pt")


@app.get("/")
def home():
    return {"message": "YOLO detection API running. Go to /docs to test it."}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = model(img)

    detections = []
    for box in results[0].boxes:
        detections.append({
            "class": model.names[int(box.cls)],
            "confidence": round(float(box.conf), 4),
            "bounding_box": [round(x, 1) for x in box.xyxy[0].tolist()]
        })

    annotated = results[0].plot()
    cv2.imwrite("last_prediction.jpg", annotated)

    return {"detections": detections, "count": len(detections)}


@app.get("/last-image")
def get_last_image():
    return FileResponse("last_prediction.jpg")