import os
import io
import math
import uuid
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
# Place your downloaded service account JSON next to main.py and name it
# serviceAccountKey.json, OR set the GOOGLE_APPLICATION_CREDENTIALS env var.
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

# Maps YOLO classification label → integer fill percentage stored in Firestore
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
    anomaly_type: str        # e.g. "broken_bin" | "blocked_road"
    reported_by: str         # worker UID or name
    sector: Optional[str] = ""
    priority: Optional[str] = "medium"  # "high" | "medium" | "low"


# ── Routing Helpers ───────────────────────────────────────────────────────────

def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Great-circle distance in km between two GPS coordinates."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lng2 - lng1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * R * math.asin(math.sqrt(a))


def _greedy_nearest_neighbor(stops: list, depot_lat: float, depot_lng: float) -> list:
    """
    Greedy Nearest-Neighbor heuristic starting from the depot.
    Each stop dict must contain 'lat' and 'lng'.
    Returns the stops in visit order (depot excluded).
    """
    unvisited = list(stops)
    ordered = []
    cur_lat, cur_lng = depot_lat, depot_lng
    while unvisited:
        nearest = min(
            unvisited,
            key=lambda s: _haversine_km(cur_lat, cur_lng, s["lat"], s["lng"]),
        )
        ordered.append(nearest)
        cur_lat, cur_lng = nearest["lat"], nearest["lng"]
        unvisited.remove(nearest)
    return ordered


def _total_route_km(ordered: list, depot_lat: float, depot_lng: float) -> float:
    """Total round-trip distance: depot → all stops → depot."""
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


# --------------------------------------------------------------------------- #
#  1. AI ANALYSIS — saves results directly to Firestore bins collection        #
# --------------------------------------------------------------------------- #

@app.post("/analyze/")
async def analyze_bin(
    image_file: UploadFile = File(...),
    bin_id: Optional[str] = None,
    lat: Optional[float] = None,
    lng: Optional[float] = None,
):
    """
    Run both YOLO models on the uploaded image and persist results to Firestore.

    Query params (all optional):
      bin_id – existing Firestore document ID to update (merge). If omitted,
               a new bin document is created and its ID is returned.
      lat, lng – GPS coordinates stored alongside the AI results.
    """
    image_bytes = await image_file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # ── Fill level ─────────────────────────────────────────────────────────
    fill_status = "Not Detected"
    fill_confidence = 0.0
    fill_results = fill_model(image)
    if len(fill_results[0].boxes) > 0:
        best = fill_results[0].boxes[0]
        fill_status = fill_model.names[int(best.cls[0])]
        fill_confidence = round(float(best.conf[0]), 3)
    fill_level_int = FILL_LEVEL_MAP.get(fill_status, 0)

    # ── Waste types ────────────────────────────────────────────────────────
    waste_results = waste_model(image)
    detected_waste = []
    for box in waste_results[0].boxes:
        wtype = waste_model.names[int(box.cls[0])]
        detected_waste.append({
            "type": wtype,
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
        if detected_waste
        else "Unknown"
    )

    # ── Firestore write ────────────────────────────────────────────────────
    bin_data: dict = {
        "fillLevel": fill_level_int,
        "wasteType": primary_waste_type,
        "aiConfidence": fill_confidence,
        "lastAnalyzed": datetime.now(timezone.utc),
    }
    if lat is not None:
        bin_data["lat"] = lat
    if lng is not None:
        bin_data["lng"] = lng

    if bin_id:
        db.collection("bins").document(bin_id).set(bin_data, merge=True)
        saved_bin_id = bin_id
    else:
        new_doc = db.collection("bins").document()
        saved_bin_id = new_doc.id
        bin_data.setdefault("lat", 0.0)
        bin_data.setdefault("lng", 0.0)
        new_doc.set(bin_data)

    return {
        "success": True,
        "bin_id": saved_bin_id,
        "filename": image_file.filename,
        "results": {
            "fill_level": {
                "status": fill_status,
                "value": fill_level_int,
                "confidence": fill_confidence,
            },
            "waste_detected": detected_waste,
        },
    }


# --------------------------------------------------------------------------- #
#  2. ROUTE OPTIMIZATION                                                        #
# --------------------------------------------------------------------------- #

@app.post("/optimize-route")
def optimize_route(req: OptimizeRouteRequest):
    """
    Fetch all bins with fillLevel > 70, compute the shortest Greedy
    Nearest-Neighbor tour, calculate fuel, and save the plan to 'routes'.
    """
    full_bins_query = db.collection("bins").where("fillLevel", ">", 70).stream()
    stops = []
    for doc in full_bins_query:
        data = doc.to_dict()
        lat = data.get("lat")
        lng = data.get("lng")
        if lat is None or lng is None:
            continue
        stops.append({
            "binId": doc.id,
            "lat": float(lat),
            "lng": float(lng),
            "fillLevel": data.get("fillLevel", 0),
            "wasteType": data.get("wasteType", "Unknown"),
            "sector": data.get("sector", ""),
        })

    if not stops:
        raise HTTPException(
            status_code=404,
            detail="No bins with fillLevel > 70 found. Nothing to optimize.",
        )

    ordered_stops = _greedy_nearest_neighbor(stops, req.depot_lat, req.depot_lng)
    total_km = _total_route_km(ordered_stops, req.depot_lat, req.depot_lng)
    # Fuel economy: 5 km per litre (as specified)
    estimated_fuel = round(total_km / 5.0, 2)

    route_id = str(uuid.uuid4())
    route_doc = {
        "routeId": route_id,
        "driverId": req.driver_id,
        "status": "active",
        "totalStops": len(ordered_stops),
        "completedStops": 0,
        "stops": ordered_stops,
        "totalDistanceKm": total_km,
        "estimatedFuel": estimated_fuel,
        "createdAt": datetime.now(timezone.utc),
    }
    db.collection("routes").document(route_id).set(route_doc)

    return {"success": True, "route": route_doc}


# --------------------------------------------------------------------------- #
#  3. WORKER SYNC                                                               #
# --------------------------------------------------------------------------- #

@app.post("/update-worker-location")
def update_worker_location(req: UpdateWorkerLocationRequest):
    """Update the live GPS position of a driver in the 'users' collection."""
    db.collection("users").document(req.driver_id).set(
        {
            "lat": req.lat,
            "lng": req.lng,
            "locationUpdatedAt": datetime.now(timezone.utc),
        },
        merge=True,
    )
    return {"success": True, "driver_id": req.driver_id, "lat": req.lat, "lng": req.lng}


@app.post("/complete-stop")
def complete_stop(req: CompleteStopRequest):
    """
    Mark one route stop as done:
    - Increment completedStops on the route.
    - Set route status to 'completed' when all stops are done.
    - Reset the collected bin's fillLevel to 0.
    """
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

    return {
        "success": True,
        "route_id": req.route_id,
        "completed_stops": new_completed,
        "total_stops": total_stops,
        "route_status": new_status,
    }


# --------------------------------------------------------------------------- #
#  4. ANOMALY REPORTING                                                         #
# --------------------------------------------------------------------------- #

@app.post("/report-anomaly")
def report_anomaly(req: ReportAnomalyRequest):
    """Log a broken bin or blocked road to the 'anomalies' collection."""
    anomaly_id = str(uuid.uuid4())
    db.collection("anomalies").document(anomaly_id).set({
        "anomalyId": anomaly_id,
        "binId": req.bin_id,
        "anomalyType": req.anomaly_type,
        "reportedBy": req.reported_by,
        "sector": req.sector,
        "priority": req.priority,
        "status": "pending",
        "createdAt": datetime.now(timezone.utc),
    })
    return {"success": True, "anomaly_id": anomaly_id}


# --------------------------------------------------------------------------- #
#  5. ADMIN KPI DASHBOARD                                                       #
# --------------------------------------------------------------------------- #

@app.get("/admin/stats")
def admin_stats():
    """
    Aggregate KPIs from all collections for the admin dashboard:
      - totalFullBins        : bins with fillLevel > 70
      - activeDrivers        : workers with status == 'active'
      - activeRoutes         : routes with status == 'active'
      - totalFuelUsedLitres  : sum of estimatedFuel on completed routes
      - pendingAnomalies     : anomalies with status == 'pending'
    """
    full_bins = list(db.collection("bins").where("fillLevel", ">", 70).stream())

    active_workers = list(
        db.collection("users")
        .where("role", "==", "worker")
        .where("status", "==", "active")
        .stream()
    )

    all_routes = [doc.to_dict() for doc in db.collection("routes").stream()]
    active_routes = [r for r in all_routes if r.get("status") == "active"]
    completed_routes = [r for r in all_routes if r.get("status") == "completed"]
    total_fuel_used = round(
        sum(r.get("estimatedFuel", 0) for r in completed_routes), 2
    )

    pending_anomalies = list(
        db.collection("anomalies").where("status", "==", "pending").stream()
    )

    return {
        "success": True,
        "stats": {
            "totalFullBins": len(full_bins),
            "activeDrivers": len(active_workers),
            "activeRoutes": len(active_routes),
            "totalFuelUsedLitres": total_fuel_used,
            "pendingAnomalies": len(pending_anomalies),
        },
    }
