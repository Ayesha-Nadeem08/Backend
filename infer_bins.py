"""
infer_bins.py — Run the two YOLO models on every image in asset/bins/
and print a JSON array of {filename, fillStatus, fillLevel, fillConf, wasteType, wasteConf}.

Usage (from Backend/):
    python infer_bins.py
"""

import json
import sys
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

FILL_LEVEL_MAP: dict[str, int] = {
    "Empty": 5, "Low": 20, "Partial": 40,
    "Half-Full": 55, "Half Full": 55, "Medium": 55,
    "Full": 85, "Overflowing": 98, "Overflow": 98, "Critical": 95,
}

def _label_to_fill(label: str) -> int:
    if label in FILL_LEVEL_MAP:
        return FILL_LEVEL_MAP[label]
    lower = label.lower().replace("_", " ").replace("-", " ")
    if "overflow" in lower or "overfull" in lower:
        return 98
    if "full" in lower and any(w in lower for w in ("half", "mid", "medium", "partial")):
        return 55
    if "full" in lower or "critical" in lower:
        return 85
    if any(w in lower for w in ("partial", "medium", "mid", "half")):
        return 55
    if "low" in lower or "quarter" in lower:
        return 20
    if "empty" in lower:
        return 5
    return 0

print("Loading models...", file=sys.stderr)
fill_model  = YOLO("models/fill_model.pt")
waste_model = YOLO("models/waste_model.pt")
print("Models loaded.", file=sys.stderr)

images_dir = Path("asset/bins")
image_files = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
print(f"Found {len(image_files)} images.", file=sys.stderr)

results = []
for img_path in image_files:
    img = Image.open(img_path).convert("RGB")

    # Fill level
    fill_status = "Not Detected"
    fill_conf   = 0.0
    fill_res = fill_model(img, conf=0.3, verbose=False)
    if fill_res[0].boxes and len(fill_res[0].boxes) > 0:
        best = fill_res[0].boxes[0]
        fill_status = fill_model.names[int(best.cls[0])].strip().title()
        fill_conf   = round(float(best.conf[0]), 3)
    fill_level = _label_to_fill(fill_status)

    # Waste type
    waste_type = "Unknown"
    waste_conf = 0.0
    if fill_status not in ("Not Detected", "Empty"):
        waste_res = waste_model(img, conf=0.3, verbose=False)
        if waste_res[0].boxes and len(waste_res[0].boxes) > 0:
            best_w = max(waste_res[0].boxes, key=lambda b: float(b.conf[0]))
            waste_type = waste_model.names[int(best_w.cls[0])]
            waste_conf = round(float(best_w.conf[0]), 3)

    results.append({
        "file":       img_path.name,
        "fillStatus": fill_status,
        "fillLevel":  fill_level,
        "fillConf":   fill_conf,
        "wasteType":  waste_type,
        "wasteConf":  waste_conf,
    })
    print(f"  {img_path.name}: {fill_status} ({fill_level}%) conf={fill_conf}  waste={waste_type} ({waste_conf})", file=sys.stderr)

print(json.dumps(results, indent=2))
