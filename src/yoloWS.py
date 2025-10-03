from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io
import torch

app = FastAPI()

# Allow cross-origin requests (so you can connect from browser or LAN)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins for testing
    allow_methods=["*"],
    allow_headers=["*"]
)

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the uploaded image
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))

        # Run YOLO
        results = model(img)

        # Convert results to JSON
        predictions = results.pandas().xyxy[0].to_dict(orient="records")
        return JSONResponse(content={"predictions": predictions})
    except Exception as e:
        return JSONResponse(content={"error": str(e)})
