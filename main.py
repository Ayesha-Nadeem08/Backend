from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
from PIL import Image
import io

# Initialize the API
app = FastAPI(title="Clean Core AI Engine", version="1.0")

# Load both models into memory when the server starts
# Make sure the paths match where you put your .pt files!
print("Loading AI Models...")
fill_model = YOLO("models/fill_model.pt")
waste_model = YOLO("models/waste_model.pt")
print("Models loaded successfully!")

@app.get("/")
def read_root():
    return {"message": "Clean Core API is running. Send a POST request to /analyze/"}

@app.post("/analyze/")
async def analyze_bin(image_file: UploadFile = File(...)):
    """
    Receives an image from the Flutter app, passes it through both YOLOv8 models,
    and returns a combined JSON report of the fill level and waste types.
    """
    # 1. Read the uploaded image
    image_bytes = await image_file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # 2. Run both models on the same image
    fill_results = fill_model(image)
    waste_results = waste_model(image)

    # 3. Extract Fill Level Data
    fill_status = "Not Detected"
    fill_confidence = 0.0
    
    # Check if the model found a bin
    if len(fill_results[0].boxes) > 0:
        # Get the box with the highest confidence
        best_fill_box = fill_results[0].boxes[0] 
        class_id = int(best_fill_box.cls[0])
        fill_status = fill_model.names[class_id]
        fill_confidence = round(float(best_fill_box.conf[0]), 3)

    # 4. Extract Waste Types Data
    detected_waste = []
    for box in waste_results[0].boxes:
        waste_class_id = int(box.cls[0])
        detected_waste.append({
            "type": waste_model.names[waste_class_id],
            "confidence": round(float(box.conf[0]), 3),
            # Sending coordinates just in case Flutter wants to draw boxes on the screen!
            "coordinates": {
                "x1": round(float(box.xyxy[0][0]), 1),
                "y1": round(float(box.xyxy[0][1]), 1),
                "x2": round(float(box.xyxy[0][2]), 1),
                "y2": round(float(box.xyxy[0][3]), 1)
            }
        })

    # 5. Return the final JSON to the Flutter App
    return {
        "success": True,
        "filename": image_file.filename,
        "results": {
            "fill_level": {
                "status": fill_status,
                "confidence": fill_confidence
            },
            "waste_detected": detected_waste
        }
    }