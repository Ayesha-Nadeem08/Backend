import os
import io
import math
import uuid
import traceback
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from ultralytics import YOLO
from PIL import Image

import firebase_admin
from firebase_admin import auth as fb_auth, credentials, firestore, messaging

from google.cloud.firestore_v1.base_query import FieldFilter

# ── Firebase Admin SDK ────────────────────────────────────────────────────────
_cred_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "serviceAccountKey.json")
cred = credentials.Certificate(_cred_path)
try:
    firebase_admin.get_app()
except ValueError:
    firebase_admin.initialize_app(cred)
db = firestore.client()

def _send_fcm(token: str, title: str, body: str):
    if not token: return
    try:
        msg = messaging.Message(
            notification=messaging.Notification(title=title, body=body),
            token=token,
        )
        messaging.send(msg)
        print(f"-> FCM Sent: {title}")
    except Exception as e:
        print(f"-> FCM Error: {e}")

# ── App + CORS ────────────────────────────────────────────────────────────────
app = FastAPI(title="Clean Core AI Engine", version="3.0")
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

FILL_LEVEL_MAP = {"Empty": 10, "Partial": 50, "Full": 95}

# Waste types that violate German noise regulations at night (22:00–07:00)
NOISY_WASTE_TYPES = {"glass", "metal"}

# ── Request / Response Schemas ────────────────────────────────────────────────
class OptimizeRouteRequest(BaseModel):
    worker_id: str           # used for area/waste-type lookup in Firestore
    depot_lat: float = 0.0
    depot_lng: float = 0.0

class UpdateWorkerLocationRequest(BaseModel):
    driver_id: str
    lat: float
    lng: float

class CompleteStopRequest(BaseModel):
    route_id: str
    bin_id: str
    worker_id: str = ""

class ReportAnomalyRequest(BaseModel):
    bin_id: str
    anomaly_type: str
    reported_by: str
    sector: Optional[str] = ""
    priority: Optional[str] = "medium"

class CreateUserRequest(BaseModel):
    email: str
    password: str = "CleanCore@123"   # default password for admin-created accounts
    first_name: str
    last_name: str
    phone: str = ""
    role: str = "worker"
    assigned_area: str = ""
    assigned_waste_type: str = ""

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

def _is_quiet_hours_berlin() -> bool:
    """True when Berlin local time (CET, UTC+1 approximation) is between 22:00 and 07:00."""
    berlin_tz = timezone(timedelta(hours=1))
    hour = datetime.now(berlin_tz).hour
    return hour >= 22 or hour < 7

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def read_root():
    return {"message": "Clean Core API v3 — German Area-Based Model. POST /optimize-route to start."}


# ═══════════════════════════════════════════════════════════════════════════════
#  1. IMAGE ANALYSIS API
# ═══════════════════════════════════════════════════════════════════════════════
@app.post("/analyze/")
async def analyze_bin(
    image_file: UploadFile = File(...),
    lat: Optional[float] = None,
    lng: Optional[float] = None,
):
    print("\n========== STARTING AI ANALYSIS ==========")
    try:
        print("Step 1: Reading uploaded image...")
        image_bytes = await image_file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        print(f"-> Image read OK ({len(image_bytes)} bytes, size={image.size})")

        print("Step 2: Running Fill Level Model...")
        fill_status = "Not Detected"
        fill_confidence = 0.0
        fill_results = fill_model(image, conf=0.5)
        if len(fill_results[0].boxes) > 0:
            best = fill_results[0].boxes[0]
            raw_label = fill_model.names[int(best.cls[0])]
            fill_status = raw_label.strip().title()
            fill_confidence = round(float(best.conf[0]), 3)
        fill_level_int = FILL_LEVEL_MAP.get(fill_status, 0)
        print(f"-> Fill Model OK: label='{fill_status}', level={fill_level_int}%, conf={fill_confidence}")

        detected_waste = []
        primary_waste_type = "Unknown"

        if fill_status not in ("Not Detected", "Empty"):
            print("Step 3: Running Waste Type Model...")
            waste_results = waste_model(image, conf=0.5)
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
        else:
            print("Step 3: Skipped — no bin detected by fill model.")

        print("Step 4: Writing to Firestore...")
        bin_data: dict = {
            "fillStatus":    fill_status,
            "fillLevel":     fill_level_int,
            "wasteType":     primary_waste_type,
            "aiConfidence":  fill_confidence,
            "detectedWaste": detected_waste,
            "lat":           lat if lat is not None else 0.0,
            "lng":           lng if lng is not None else 0.0,
            "isLocked":      False,
            "lastAnalyzed":  datetime.now(timezone.utc),
        }

        new_doc = db.collection("bins").document()
        saved_bin_id = new_doc.id
        new_doc.set(bin_data)
        print(f"-> Created new bin: {saved_bin_id}")

        # --- SFR-017: FCM Push Notifications for Urgent Overflows ---
        if fill_level_int >= 90:
            # Find an active route to dynamically append this urgent pickup
            active_routes = list(db.collection("routes").where(filter=FieldFilter("status", "==", "active")).limit(1).stream())
            if active_routes:
                route_doc = active_routes[0]
                r_data = route_doc.to_dict()
                stops = r_data.get("stops", [])
                
                # Assign this bin to the found route
                bin_area = r_data.get("assignedArea", "Unknown")
                
                # We need to make sure we don't duplicate
                if not any(s.get("binId") == saved_bin_id for s in stops):
                    new_doc.update({"isLocked": True, "routeId": route_doc.id, "area": bin_area})
                    
                    stops.append({
                        "binId": saved_bin_id,
                        "lat": lat if lat is not None else 0.0,
                        "lng": lng if lng is not None else 0.0,
                        "fillLevel": fill_level_int,
                        "wasteType": primary_waste_type,
                        "area": bin_area,
                    })
                    route_doc.reference.update({
                        "stops": stops,
                        "totalStops": r_data.get("totalStops", 0) + 1
                    })
                    
                    # Notify the worker handling this route
                    worker_id = r_data.get("driverId")
                    if worker_id:
                        worker_snap = db.collection("users").document(worker_id).get()
                        if worker_snap.exists:
                            fcm_token = worker_snap.to_dict().get("fcmToken")
                            if fcm_token:
                                _send_fcm(
                                    token=fcm_token,
                                    title="🚨 Urgent Pickup Added",
                                    body=f"A critically full bin in {bin_area} has been added to your route!"
                                )

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
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}")


# ═══════════════════════════════════════════════════════════════════════════════
#  2. ROUTING API — German Area-Based Assignment Model
# ═══════════════════════════════════════════════════════════════════════════════
#
#  Worker profile (users/{worker_id}) must have:
#    assignedArea      (str) — e.g. "Mitte", "Pankow"
#    assignedWasteType (str) — e.g. "Bio", "Paper", "Plastic"
#
#  Bin documents (bins/{bin_id}) must have:
#    area (str) — matches worker's assignedArea
#    wasteType (str) — matches worker's assignedWasteType
#    fillLevel (int) — threshold > 70 to qualify
#    isLocked (bool) — True if already assigned to another active route
#
#  German Standard constraints applied:
#    • Noise regulation: Glass/Metal bins skipped between 22:00–07:00 Berlin time
#    • Carbon footprint: totalDistanceKm * 0.95 kg CO2
#    • Concurrency lock: bins set isLocked=True for the route duration
# ═══════════════════════════════════════════════════════════════════════════════
@app.post("/optimize-route")
def optimize_route(req: OptimizeRouteRequest):
    # 1. Load worker profile to get area + waste type assignments
    worker_snap = db.collection("users").document(req.worker_id).get()
    if not worker_snap.exists:
        raise HTTPException(status_code=404, detail=f"Worker '{req.worker_id}' not found.")

    worker = worker_snap.to_dict()
    assigned_area  = worker.get("assignedArea", "")
    assigned_waste = worker.get("assignedWasteType", "").lower()
    quiet_hours    = _is_quiet_hours_berlin()

    # 2. Fetch full bins and apply German-standard filters in Python
    bin_query = db.collection("bins").where(filter=FieldFilter("fillLevel", ">", 70)).stream()
    available_stops = []

    for doc in bin_query:
        data = doc.to_dict()

        # Concurrency: skip bins already locked by another active route
        if data.get("isLocked", False):
            continue

        # Area filter: worker only serves their assigned district
        bin_area = data.get("area", data.get("sector", ""))
        if assigned_area and bin_area != assigned_area:
            continue

        # Waste type filter: worker only handles their assigned stream
        waste_type = data.get("wasteType", "")
        if assigned_waste and waste_type.lower() != assigned_waste:
            continue

        # German noise regulation: no Glass or Metal collections at night
        if quiet_hours and waste_type.lower() in NOISY_WASTE_TYPES:
            continue

        lat = float(data.get("lat", 0))
        lng = float(data.get("lng", 0))
        if lat == 0.0 and lng == 0.0:
            continue

        available_stops.append({
            "binId":     doc.id,
            "lat":       lat,
            "lng":       lng,
            "fillLevel": data.get("fillLevel", 0),
            "wasteType": waste_type,
            "area":      bin_area,
        })

    if not available_stops:
        detail = "No available bins match the worker's area and waste type constraints."
        if quiet_hours:
            detail += " Note: Glass/Metal bins are excluded during quiet hours (22:00–07:00 Berlin time)."
        raise HTTPException(status_code=404, detail=detail)

    # 3. Greedy Nearest-Neighbor ordering
    ordered_stops = _greedy_nearest_neighbor(available_stops, req.depot_lat, req.depot_lng)

    # 4. Distance, fuel, and CO2 footprint (German standard: 0.95 kg CO2/km)
    total_km   = _total_route_km(ordered_stops, req.depot_lat, req.depot_lng)
    fuel       = round(total_km / 5.0, 2)
    carbon_kg  = round(total_km * 0.95, 3)

    # 5. Save route document
    route_id = str(uuid.uuid4())
    route_doc = {
        "routeId":           route_id,
        "workerId":          req.worker_id,
        "driverId":          req.worker_id,   # kept for Flutter backward compat
        "assignedArea":      assigned_area,
        "assignedWasteType": worker.get("assignedWasteType", ""),
        "status":            "active",
        "stops":             ordered_stops,
        "totalStops":        len(ordered_stops),
        "completedStops":    0,
        "totalDistanceKm":   total_km,
        "estimatedFuel":     fuel,
        "carbonFootprintKg": carbon_kg,
        "createdAt":         datetime.now(timezone.utc),
    }
    db.collection("routes").document(route_id).set(route_doc)

    # 6. Lock every bin atomically so other workers cannot pick them up
    batch = db.batch()
    for stop in ordered_stops:
        bin_ref = db.collection("bins").document(stop["binId"])
        batch.update(bin_ref, {"isLocked": True, "routeId": route_id})
    batch.commit()

    return {"success": True, "route": route_doc}


# ── Worker Route Query ────────────────────────────────────────────────────────
@app.get("/worker/my-route/{worker_id}")
def get_worker_route(worker_id: str):
    """Returns the worker's current active route with all ordered stops."""
    routes = list(
        db.collection("routes")
        .where("driverId", "==", worker_id)
        .where("status", "==", "active")
        .limit(1)
        .stream()
    )
    if not routes:
        raise HTTPException(status_code=404, detail="No active route found for this worker.")
    return {"success": True, "route": routes[0].to_dict()}


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
    now_utc    = datetime.now(timezone.utc)
    route_ref  = db.collection("routes").document(req.route_id)
    route_snap = route_ref.get()
    if not route_snap.exists:
        raise HTTPException(status_code=404, detail=f"Route '{req.route_id}' not found.")

    route_data = route_snap.to_dict()

    # Mark the individual stop completed and capture its metadata for the log
    stops = list(route_data.get("stops", []))
    fill_at_pickup, bin_area, bin_waste_type = 0, "", ""
    bin_lat, bin_lng = 0.0, 0.0
    for stop in stops:
        if stop.get("binId") == req.bin_id:
            stop["completed"]  = True
            fill_at_pickup     = stop.get("fillLevel", 0)
            bin_area           = stop.get("area", "")
            bin_waste_type     = stop.get("wasteType", "")
            bin_lat            = float(stop.get("lat", 0))
            bin_lng            = float(stop.get("lng", 0))
            break

    new_completed = route_data.get("completedStops", 0) + 1
    total_stops   = route_data.get("totalStops", 0)
    new_status    = "completed" if new_completed >= total_stops else "active"

    update_payload: dict = {
        "completedStops": new_completed,
        "status":         new_status,
        "stops":          stops,
    }
    if new_status == "completed":
        update_payload["completedAt"] = now_utc
    route_ref.update(update_payload)

    # Unlock the bin and reset fill level
    db.collection("bins").document(req.bin_id).update({
        "fillLevel":     0,
        "fillStatus":    "Empty",
        "isLocked":      False,
        "routeId":       firestore.DELETE_FIELD,
        "lastCollected": now_utc,
    })

    # Write pickup log for analytics — drives the Collections page
    log_id = str(uuid.uuid4())
    db.collection("pickupLogs").document(log_id).set({
        "logId":             log_id,
        "workerId":          req.worker_id,
        "routeId":           req.route_id,
        "binId":             req.bin_id,
        "area":              bin_area,
        "wasteType":         bin_waste_type,
        "fillLevelAtPickup": fill_at_pickup,
        "binLat":            bin_lat,
        "binLng":            bin_lng,
        "completedAt":       now_utc,
    })

    return {
        "success":         True,
        "route_status":    new_status,
        "completed_stops": new_completed,
        "total_stops":     total_stops,
    }


# ── Admin: pickup logs ────────────────────────────────────────────────────────
@app.get("/admin/pickup-logs")
def admin_pickup_logs(worker_id: Optional[str] = None, limit: int = 100):
    limit = min(limit, 500)
    if worker_id:
        q = (
            db.collection("pickupLogs")
            .where(filter=FieldFilter("workerId", "==", worker_id))
            .order_by("completedAt", direction=firestore.Query.DESCENDING)
            .limit(limit)
        )
    else:
        q = (
            db.collection("pickupLogs")
            .order_by("completedAt", direction=firestore.Query.DESCENDING)
            .limit(limit)
        )
    logs = [doc.to_dict() for doc in q.stream()]
    return {"success": True, "logs": logs, "count": len(logs)}


# ── Admin: all routes ─────────────────────────────────────────────────────────
@app.get("/admin/routes")
def admin_routes_list(status: Optional[str] = None, limit: int = 50):
    limit = min(limit, 200)
    if status:
        q = (
            db.collection("routes")
            .where(filter=FieldFilter("status", "==", status))
            .order_by("createdAt", direction=firestore.Query.DESCENDING)
            .limit(limit)
        )
    else:
        q = (
            db.collection("routes")
            .order_by("createdAt", direction=firestore.Query.DESCENDING)
            .limit(limit)
        )
    routes = [doc.to_dict() for doc in q.stream()]
    return {"success": True, "routes": routes, "count": len(routes)}


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


# ── Admin: Create user (Auth + Firestore) ─────────────────────────────────────
@app.post("/admin/create-user")
def admin_create_user(req: CreateUserRequest):
    """Creates a Firebase Auth account AND the corresponding Firestore user document."""
    display_name = f"{req.first_name} {req.last_name}".strip()
    try:
        user_record = fb_auth.create_user(
            email=req.email,
            password=req.password,
            display_name=display_name,
            email_verified=True,
        )
    except fb_auth.EmailAlreadyExistsError:
        raise HTTPException(status_code=409, detail=f"Email '{req.email}' is already registered.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Auth creation failed: {e}")

    uid = user_record.uid
    db.collection("users").document(uid).set({
        "uid":               uid,
        "email":             req.email,
        "firstName":         req.first_name,
        "lastName":          req.last_name,
        "phoneNumber":       req.phone,
        "role":              req.role,
        "status":            "active",
        "collections":       0,
        "routes":            0,
        "profilePicture":    "",
        "assignedArea":      req.assigned_area if req.role == "worker" else "",
        "assignedWasteType": req.assigned_waste_type if req.role == "worker" else "",
        "createdAt":         datetime.now(timezone.utc),
    })
    return {"success": True, "uid": uid, "email": req.email}


# ── Admin: Delete user (Auth + Firestore) ─────────────────────────────────────
@app.delete("/admin/delete-user/{uid}")
def admin_delete_user(uid: str):
    """Deletes both the Firebase Auth account and the Firestore user document."""
    # 1. Delete Firestore document
    db.collection("users").document(uid).delete()

    # 2. Delete Firebase Auth account
    try:
        fb_auth.delete_user(uid)
    except fb_auth.UserNotFoundError:
        pass   # Auth account already gone — not an error
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Auth deletion failed: {e}")

    return {"success": True, "uid": uid}


# ── Admin KPI Dashboard ───────────────────────────────────────────────────────
@app.get("/admin/stats")
def admin_stats():
    full_bins = list(db.collection("bins").where(filter=FieldFilter("fillLevel", ">", 70)).stream())
    active_workers = list(
        db.collection("users")
        .where(filter=FieldFilter("role", "==", "worker"))
        .where(filter=FieldFilter("status", "==", "active"))
        .stream()
    )
    all_routes       = [doc.to_dict() for doc in db.collection("routes").stream()]
    active_routes    = [r for r in all_routes if r.get("status") == "active"]
    completed_routes = [r for r in all_routes if r.get("status") == "completed"]

    total_fuel   = round(sum(r.get("estimatedFuel", 0)     for r in completed_routes), 2)
    total_carbon = round(sum(r.get("carbonFootprintKg", 0) for r in completed_routes), 3)

    total_stops = sum(r.get("totalStops", 0)     for r in completed_routes)
    done_stops  = sum(r.get("completedStops", 0) for r in completed_routes)
    efficiency  = round(done_stops / total_stops * 100, 1) if total_stops > 0 else 0.0

    pending_anomalies = list(db.collection("anomalies").where("status", "==", "pending").stream())

    return {
        "success": True,
        "stats": {
            "totalFullBins":       len(full_bins),
            "activeDrivers":       len(active_workers),
            "activeRoutes":        len(active_routes),
            "totalFuelUsedLitres": total_fuel,
            "totalCarbonKg":       total_carbon,
            "fleetEfficiency":     efficiency,
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

# ── Serve Admin Panel (Vite build) ────────────────────────────────────────────
# The Admin Panel React app is built into ../Admin_Panel/dist with base="/admin/".
# Static assets (JS, CSS, images) are served at /admin/assets/...,
# and the SPA catch-all returns index.html for all other /admin/* paths.
_admin_dist = Path(__file__).resolve().parent.parent / "Admin_Panel" / "dist"

if _admin_dist.is_dir():
    # Serve JS/CSS bundles under /admin/assets
    _assets_dir = _admin_dist / "assets"
    if _assets_dir.is_dir():
        app.mount("/admin/assets", StaticFiles(directory=str(_assets_dir)), name="admin_assets")

    @app.get("/admin/{rest_of_path:path}", response_class=HTMLResponse)
    def serve_admin_spa(rest_of_path: str = ""):
        """SPA catch-all: serve static file if it exists, else index.html."""
        # Check if a real file is being requested (e.g. favicon.ico)
        candidate = _admin_dist / rest_of_path
        if rest_of_path and candidate.is_file():
            return HTMLResponse(
                content=candidate.read_bytes(),
                media_type="application/octet-stream",
            )
        # Otherwise serve index.html for client-side routing
        index_file = _admin_dist / "index.html"
        if not index_file.exists():
            raise HTTPException(status_code=404, detail="Admin panel not built. Run 'npm run build' in Admin_Panel.")
        return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
else:
    print(f"⚠ Admin Panel dist not found at {_admin_dist}. Run 'npm run build' in Admin_Panel/")
