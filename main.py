import os
import io
import math
import uuid
import traceback
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO
from PIL import Image

import firebase_admin
from firebase_admin import credentials, firestore

# ── Firebase Admin SDK ────────────────────────────────────────────────────────
_cred_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "serviceAccountKey.json")
cred = credentials.Certificate(_cred_path)
firebase_admin.initialize_app(cred)
db = firestore.client()

# ── App + CORS ────────────────────────────────────────────────────────────────
app = FastAPI(title="Clean Core AI Engine", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── YOLO Models ───────────────────────────────────────────────────────────────
print("Loading AI Models...")
fill_model = YOLO("models/fill_model.pt")
waste_model = YOLO("models/waste_model.pt")
print("Models loaded successfully!")

# Maps YOLO classification label → integer fill percentage.
# Used by the Routing API to query bins where fillLevel > 70.
FILL_LEVEL_MAP = {"Empty": 10, "Partial": 50, "Full": 95}

# ── Request / Response Schemas ────────────────────────────────────────────────
class OptimizeRouteRequest(BaseModel):
    driver_id: str
    depot_lat: float = 0.0
    depot_lng: float = 0.0

class UpdateWorkerLocationRequest(BaseModel):
    driver_id: str
    lat: float
    lng: float

class CompleteStopRequest(BaseModel):
    route_id: str
    bin_id: str

class ReportAnomalyRequest(BaseModel):
    bin_id: str
    anomaly_type: str
    reported_by: str
    sector: Optional[str] = ""
    priority: Optional[str] = "medium"

# ── Routing Helpers ───────────────────────────────────────────────────────────
def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lng2 - lng1)
    a = (math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))

def _greedy_nearest_neighbor(stops: list, depot_lat: float, depot_lng: float) -> list:
    unvisited = list(stops)
    ordered = []
    cur_lat, cur_lng = depot_lat, depot_lng
    while unvisited:
        nearest = min(unvisited, key=lambda s: _haversine_km(cur_lat, cur_lng, s["lat"], s["lng"]))
        ordered.append(nearest)
        cur_lat, cur_lng = nearest["lat"], nearest["lng"]
        unvisited.remove(nearest)
    return ordered

def _total_route_km(ordered: list, depot_lat: float, depot_lng: float) -> float:
    total = 0.0
    prev_lat, prev_lng = depot_lat, depot_lng
    for stop in ordered:
        total += _haversine_km(prev_lat, prev_lng, stop["lat"], stop["lng"])
        prev_lat, prev_lng = stop["lat"], stop["lng"]
    total += _haversine_km(prev_lat, prev_lng, depot_lat, depot_lng)
    return round(total, 3)

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def read_root():
    return {"message": "Clean Core API v2 is running. POST to /analyze/ to start."}


# ═══════════════════════════════════════════════════════════════════════════════
#  1. IMAGE ANALYSIS API
# ═══════════════════════════════════════════════════════════════════════════════
#
#  HOW IT WORKS:
#  ─────────────
#  The mobile app (CleanCore) calls POST /analyze/ with two things:
#    • image_file  — a photo of the bin taken from the phone camera
#    • lat, lng    — the GPS coordinates of that bin at the moment of the photo
#
#  The lat/lng are NEVER processed or analysed by this API.
#  They are received exactly as sent by the phone and stored in Firestore
#  alongside the AI result, unchanged.
#
#  STEP 1 — Image only goes through two YOLO models:
#    fill_model  → classifies fill level:  "Empty" | "Partial" | "Full"
#    waste_model → detects waste objects:  "plastic" | "organic" | etc.
#
#  STEP 2 — Raw AI output is captured exactly as produced by YOLO:
#    fillStatus (str)   — raw label from fill_model, e.g. "Full"
#    fillLevel  (int)   — integer representation (Empty→10, Partial→50, Full→95)
#                         needed so the Routing API can query  fillLevel > 70
#    wasteType  (str)   — highest-confidence waste label from waste_model
#    aiConfidence (float) — confidence score of the fill prediction (0.0–1.0)
#    detectedWaste (list) — all detected waste objects with their confidence
#                           scores and bounding box coordinates
#
#  STEP 3 — One Firestore document is written to the "bins" collection:
#
#    bins/{bin_id}  (merge=True so existing fields are preserved):
#    {
#      "fillStatus":     "Full",          ← raw YOLO label, stored as-is
#      "fillLevel":      95,              ← integer for routing threshold queries
#      "wasteType":      "plastic",       ← primary waste label from waste_model
#      "aiConfidence":   0.934,           ← fill model confidence
#      "detectedWaste":  [                ← complete raw waste detection list
#                          { "type": "plastic",  "confidence": 0.934,
#                            "coordinates": {"x1":12,"y1":8,"x2":200,"y2":190} },
#                          { "type": "organic",  "confidence": 0.621, ... }
#                        ],
#      "lat":            33.6844,         ← received from phone, stored as-is
#      "lng":            73.0479,         ← received from phone, stored as-is
#      "lastAnalyzed":   <timestamp>
#    }
#
#  MOBILE APP INTEGRATION:
#    ApiService.analyzeBin(imageFile, binId, lat, lng)
#    → POST /analyze/?bin_id=BIN001&lat=33.6844&lng=73.0479
#    → multipart body: image_file = <photo bytes>
#    → returns { success, bin_id, results: { fill_level, waste_detected } }
#
#  ADMIN PANEL INTEGRATION:
#    Reads the same bins/{bin_id} document to display fill status on the map.
#    The Routing API (below) queries fillLevel > 70 to find bins needing pickup.
# ═══════════════════════════════════════════════════════════════════════════════
@app.post("/analyze/")
async def analyze_bin(
    image_file: UploadFile = File(...),
    bin_id: Optional[str] = None,
    lat: Optional[float] = None,
    lng: Optional[float] = None,
):
    print("\n========== STARTING AI ANALYSIS ==========")
    try:
        # ── Step 1: Read image (location params are not touched here) ──────────
        print("Step 1: Reading uploaded image...")
        image_bytes = await image_file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        print(f"-> Image read OK ({len(image_bytes)} bytes, size={image.size})")

        # ── Step 2: Run Fill Level Model ───────────────────────────────────────
        print("Step 2: Running Fill Level Model...")
        fill_status = "Not Detected"
        fill_confidence = 0.0
        fill_results = fill_model(image)
        if len(fill_results[0].boxes) > 0:
            best = fill_results[0].boxes[0]
            fill_status = fill_model.names[int(best.cls[0])]   # raw label
            fill_confidence = round(float(best.conf[0]), 3)
        fill_level_int = FILL_LEVEL_MAP.get(fill_status, 0)   # integer for queries
        print(f"-> Fill Model OK: label='{fill_status}', level={fill_level_int}%, conf={fill_confidence}")

        # ── Step 3: Run Waste Type Model ───────────────────────────────────────
        print("Step 3: Running Waste Type Model...")
        waste_results = waste_model(image)
        detected_waste = []
        for box in waste_results[0].boxes:
            detected_waste.append({
                "type":       waste_model.names[int(box.cls[0])],
                "confidence": round(float(box.conf[0]), 3),
                "coordinates": {
                    "x1": round(float(box.xyxy[0][0]), 1),
                    "y1": round(float(box.xyxy[0][1]), 1),
                    "x2": round(float(box.xyxy[0][2]), 1),
                    "y2": round(float(box.xyxy[0][3]), 1),
                },
            })
        primary_waste_type = (
            max(detected_waste, key=lambda w: w["confidence"])["type"]
            if detected_waste else "Unknown"
        )
        print(f"-> Waste Model OK: primary='{primary_waste_type}', detections={len(detected_waste)}")

        # ── Step 4: Build Firestore document ───────────────────────────────────
        # lat and lng come directly from the request parameters — no modification.
        # fill_status is the raw YOLO label — stored as-is.
        # fill_level_int is derived only so the Routing API can use fillLevel > 70.
        print("Step 4: Writing to Firestore...")
        bin_data: dict = {
            "fillStatus":    fill_status,       # raw YOLO label: "Empty"|"Partial"|"Full"
            "fillLevel":     fill_level_int,     # integer: 10 | 50 | 95
            "wasteType":     primary_waste_type,
            "aiConfidence":  fill_confidence,
            "detectedWaste": detected_waste,     # full raw detection list
            "lastAnalyzed":  datetime.now(timezone.utc),
        }

        # Store lat/lng exactly as received from the phone — no processing.
        if lat is not None:
            bin_data["lat"] = lat
        if lng is not None:
            bin_data["lng"] = lng

        if bin_id:
            db.collection("bins").document(bin_id).set(bin_data, merge=True)
            saved_bin_id = bin_id
            print(f"-> Updated existing bin: {bin_id}")
        else:
            new_doc = db.collection("bins").document()
            saved_bin_id = new_doc.id
            bin_data.setdefault("lat", 0.0)
            bin_data.setdefault("lng", 0.0)
            new_doc.set(bin_data)
            print(f"-> Created new bin: {saved_bin_id}")

        print("========== ANALYSIS COMPLETE ==========\n")
        return {
            "success": True,
            "bin_id": saved_bin_id,
            "filename": image_file.filename,
            "results": {
                "fill_level": {
                    "status":     fill_status,
                    "value":      fill_level_int,
                    "confidence": fill_confidence,
                },
                "waste_detected": detected_waste,
            },
        }

    except Exception as e:
        print("\n!!!!! SERVER CRASH DETECTED !!!!!")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {str(e)}")
        traceback.print_exc()
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}")


# ═══════════════════════════════════════════════════════════════════════════════
#  2. ROUTING API
# ═══════════════════════════════════════════════════════════════════════════════
#
#  HOW IT WORKS — END TO END:
#  ──────────────────────────
#  The Admin Panel calls POST /optimize-route once per driver shift to generate
#  a collection plan. Here is the exact sequence:
#
#  ── INPUT ─────────────────────────────────────────────────────────────────────
#  Request body (JSON):
#    {
#      "driver_id":  "UID_of_the_driver",   ← Firestore UID from users collection
#      "depot_lat":  33.6844,               ← GPS of the depot/garage (start point)
#      "depot_lng":  73.0479
#    }
#
#  ── STEP 1: FIND BINS THAT NEED COLLECTION ────────────────────────────────────
#  Queries Firestore:
#    db.collection("bins").where("fillLevel", ">", 70)
#
#  Each matching bin document has this shape:
#    bins/{bin_id}:
#    {
#      "fillStatus":  "Full",      ← written by /analyze/
#      "fillLevel":   95,          ← written by /analyze/ — used for this query
#      "wasteType":   "plastic",
#      "lat":         33.6844,     ← GPS coordinates of the physical bin
#      "lng":         73.0479,
#      "sector":      "Zone-A"     ← optional admin label
#    }
#
#  Only bins that have both lat and lng are included (bins without GPS are skipped).
#  Each qualifying bin becomes a "stop" dict:
#    {
#      "binId":     "abc123",
#      "lat":       33.6844,
#      "lng":       73.0479,
#      "fillLevel": 95,
#      "wasteType": "plastic",
#      "sector":    "Zone-A"
#    }
#
#  ── STEP 2: ORDER THE STOPS — GREEDY NEAREST-NEIGHBOR ────────────────────────
#  Starting from the depot coordinates, the algorithm repeatedly picks the
#  closest unvisited bin using the Haversine great-circle distance formula.
#
#  Example with 3 bins (A, B, C) and depot D:
#    D → nearest to D = A  →  nearest to A = C  →  nearest to C = B
#  Result: ordered_stops = [A, C, B]
#
#  This is a O(n²) heuristic — fast enough for typical city-scale deployments
#  (hundreds of bins) and produces routes within ~20% of optimal.
#
#  ── STEP 3: CALCULATE DISTANCE & FUEL ────────────────────────────────────────
#  Total route km = D→A + A→C + C→B + B→D  (round trip back to depot)
#  Estimated fuel = total_km / 5.0  (5 km per litre assumption)
#
#  ── STEP 4: WRITE ROUTE TO FIRESTORE ─────────────────────────────────────────
#  A new document is created in the "routes" collection:
#
#    routes/{route_id}:
#    {
#      "routeId":         "uuid-v4-string",
#      "driverId":        "UID_of_the_driver",    ← links to users/{driverId}
#      "status":          "active",               ← "active" | "completed"
#      "totalStops":      3,
#      "completedStops":  0,                      ← incremented by /complete-stop
#      "stops": [                                 ← ordered list of stops
#        { "binId":"A", "lat":33.68, "lng":73.04, "fillLevel":95, "wasteType":"plastic", "sector":"Zone-A" },
#        { "binId":"C", "lat":33.71, "lng":73.06, "fillLevel":80, "wasteType":"organic", "sector":"Zone-B" },
#        { "binId":"B", "lat":33.69, "lng":73.05, "fillLevel":75, "wasteType":"mixed",   "sector":"Zone-A" }
#      ],
#      "totalDistanceKm": 12.4,
#      "estimatedFuel":   2.48,
#      "createdAt":       <timestamp>
#    }
#
#  ── MOBILE APP INTEGRATION ────────────────────────────────────────────────────
#  The driver app reads routes where driverId == currentUser.uid and status == "active".
#  It renders the ordered stops[] list as waypoints on the map.
#  When the driver arrives at a bin and presses "Mark Complete", the app calls
#  POST /complete-stop which:
#    • increments completedStops on the route document
#    • sets the bin's fillLevel back to 0 in the bins collection
#    • sets route status to "completed" when completedStops >= totalStops
#
#  ── ADMIN PANEL INTEGRATION ───────────────────────────────────────────────────
#  Admin panel calls /optimize-route to generate the route.
#  Admin panel reads routes collection to display active routes on the map,
#  shows completedStops / totalStops progress, and totalDistanceKm / estimatedFuel
#  in the KPI dashboard (/admin/stats).
#
#  ── ERROR CASE ────────────────────────────────────────────────────────────────
#  Returns HTTP 404 if no bin has fillLevel > 70 (nothing to collect).
# ═══════════════════════════════════════════════════════════════════════════════
@app.post("/optimize-route")
def optimize_route(req: OptimizeRouteRequest):
    full_bins_query = db.collection("bins").where("fillLevel", ">", 70).stream()
    stops = []
    for doc in full_bins_query:
        data = doc.to_dict()
        lat = data.get("lat")
        lng = data.get("lng")
        if lat is None or lng is None:
            continue
        stops.append({
            "binId":     doc.id,
            "lat":       float(lat),
            "lng":       float(lng),
            "fillLevel": data.get("fillLevel", 0),
            "wasteType": data.get("wasteType", "Unknown"),
            "sector":    data.get("sector", ""),
        })

    if not stops:
        raise HTTPException(status_code=404, detail="No bins with fillLevel > 70 found.")

    ordered_stops = _greedy_nearest_neighbor(stops, req.depot_lat, req.depot_lng)
    total_km = _total_route_km(ordered_stops, req.depot_lat, req.depot_lng)
    estimated_fuel = round(total_km / 5.0, 2)

    route_id = str(uuid.uuid4())
    route_doc = {
        "routeId":        route_id,
        "driverId":       req.driver_id,
        "status":         "active",
        "totalStops":     len(ordered_stops),
        "completedStops": 0,
        "stops":          ordered_stops,
        "totalDistanceKm": total_km,
        "estimatedFuel":  estimated_fuel,
        "createdAt":      datetime.now(timezone.utc),
    }
    db.collection("routes").document(route_id).set(route_doc)
    return {"success": True, "route": route_doc}


# ── Worker Sync ───────────────────────────────────────────────────────────────
@app.post("/update-worker-location")
def update_worker_location(req: UpdateWorkerLocationRequest):
    db.collection("users").document(req.driver_id).set(
        {"lat": req.lat, "lng": req.lng, "locationUpdatedAt": datetime.now(timezone.utc)},
        merge=True,
    )
    return {"success": True, "driver_id": req.driver_id, "lat": req.lat, "lng": req.lng}

@app.post("/complete-stop")
def complete_stop(req: CompleteStopRequest):
    route_ref = db.collection("routes").document(req.route_id)
    route_snap = route_ref.get()
    if not route_snap.exists:
        raise HTTPException(status_code=404, detail=f"Route '{req.route_id}' not found.")

    route_data = route_snap.to_dict()
    new_completed = route_data.get("completedStops", 0) + 1
    total_stops = route_data.get("totalStops", 0)
    new_status = "completed" if new_completed >= total_stops else "active"

    update_payload = {"completedStops": new_completed, "status": new_status}
    if new_status == "completed":
        update_payload["completedAt"] = datetime.now(timezone.utc)
    route_ref.update(update_payload)

    db.collection("bins").document(req.bin_id).update({
        "fillLevel": 0,
        "lastCollected": datetime.now(timezone.utc),
    })
    return {"success": True, "route_status": new_status}

@app.post("/report-anomaly")
def report_anomaly(req: ReportAnomalyRequest):
    anomaly_id = str(uuid.uuid4())
    db.collection("anomalies").document(anomaly_id).set({
        "anomalyId":    anomaly_id,
        "binId":        req.bin_id,
        "anomalyType":  req.anomaly_type,
        "reportedBy":   req.reported_by,
        "sector":       req.sector,
        "priority":     req.priority,
        "status":       "pending",
        "createdAt":    datetime.now(timezone.utc),
    })
    return {"success": True, "anomaly_id": anomaly_id}


# ── Admin KPI Dashboard ───────────────────────────────────────────────────────
@app.get("/admin/stats")
def admin_stats():
    full_bins = list(db.collection("bins").where("fillLevel", ">", 70).stream())
    active_workers = list(
        db.collection("users")
        .where("role", "==", "worker")
        .where("status", "==", "active")
        .stream()
    )
    all_routes = [doc.to_dict() for doc in db.collection("routes").stream()]
    active_routes    = [r for r in all_routes if r.get("status") == "active"]
    completed_routes = [r for r in all_routes if r.get("status") == "completed"]
    total_fuel_used  = round(sum(r.get("estimatedFuel", 0) for r in completed_routes), 2)
    pending_anomalies = list(db.collection("anomalies").where("status", "==", "pending").stream())

    return {
        "success": True,
        "stats": {
            "totalFullBins":       len(full_bins),
            "activeDrivers":       len(active_workers),
            "activeRoutes":        len(active_routes),
            "totalFuelUsedLitres": total_fuel_used,
            "pendingAnomalies":    len(pending_anomalies),
        },
    }


# ── Debug / Step-by-step test endpoint ───────────────────────────────────────
@app.post("/test-image/")
async def test_image(image_file: UploadFile = File(...)):
    """Isolates each processing step to identify failures."""
    results = {}
    try:
        image_bytes = await image_file.read()
        results["step1_read"] = f"OK — {len(image_bytes)} bytes"
    except Exception as e:
        return {"failed_at": "step1_read", "error": str(e)}
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        results["step2_pil"] = f"OK — {image.size}"
    except Exception as e:
        return {"failed_at": "step2_pil", "error": str(e)}
    try:
        fill_results = fill_model(image)
        results["step3_fill_model"] = f"OK — {len(fill_results[0].boxes)} boxes"
    except Exception as e:
        return {"failed_at": "step3_fill_model", "error": str(e)}
    try:
        waste_results = waste_model(image)
        results["step4_waste_model"] = f"OK — {len(waste_results[0].boxes)} boxes"
    except Exception as e:
        return {"failed_at": "step4_waste_model", "error": str(e)}
    try:
        db.collection("bins").document("__test__").set({"ping": True}, merge=True)
        results["step5_firestore"] = "OK"
    except Exception as e:
        return {"failed_at": "step5_firestore", "error": str(e)}
    return {"all_steps": "passed", "details": results}
