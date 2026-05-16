import os
import io
import base64
import math
import uuid
import traceback
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from ultralytics import YOLO
from PIL import Image

import firebase_admin
from firebase_admin import auth as fb_auth, credentials, firestore, messaging

from google.cloud.firestore_v1.base_query import FieldFilter

import torch
from ultralytics.nn.tasks import DetectionModel

torch.serialization.add_safe_globals([DetectionModel])

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
_models: dict = {}

def _load_models() -> None:
    global _models
    print("Loading AI models...")
    _orig = torch.load
    torch.load = lambda *a, **kw: _orig(*a, **{**kw, "weights_only": False})
    industrial = Path("models/fill_model_industrial.pt")
    _models["fill"]  = YOLO(str(industrial) if industrial.exists() else "models/fill_model.pt")
    _models["waste"] = YOLO("models/waste_model.pt")
    torch.load = _orig
    source = "industrial" if industrial.exists() else "consumer"
    print(f"Models loaded — fill: {source}, waste: ready")

_load_models()

FILL_LEVEL_MAP: dict[str, int] = {
    # ── Exact labels (as returned by model after .strip().title()) ────────────
    "Empty":       5,
    "Low":         20,
    "Partial":     40,
    "Half-Full":   55,
    "Half Full":   55,
    "Medium":      55,
    "Full":        85,
    "Overflowing": 98,
    "Overflow":    98,
    "Critical":    95,
    # ── Fallbacks for raw lowercase / uppercase variants ──────────────────────
    "empty":       5,
    "low":         20,
    "partial":     40,
    "half-full":   55,
    "half full":   55,
    "medium":      55,
    "full":        85,
    "overflowing": 98,
    "overflow":    98,
    "critical":    95,
}

def _label_to_fill_int(label: str) -> int:
    """Map a YOLO fill-level class label to a 0-100 integer.

    Tries an exact lookup first, then a case-insensitive keyword scan so that
    unexpected label variants (e.g. 'Half_Full', 'FULL', 'Near Full') never
    silently produce 0.
    """
    if label in FILL_LEVEL_MAP:
        return FILL_LEVEL_MAP[label]
    lower = label.lower().replace("_", " ").replace("-", " ")
    if "overflow" in lower or "overfull" in lower:
        return 98
    if "full" in lower and ("half" in lower or "mid" in lower or "medium" in lower or "partial" in lower):
        return 55
    if "full" in lower or "critical" in lower:
        return 85
    if "partial" in lower or "medium" in lower or "mid" in lower or "half" in lower:
        return 55
    if "low" in lower or "quarter" in lower:
        return 20
    if "empty" in lower:
        return 5
    return 0   # truly unknown

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

class UpdateUserRequest(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    phone: Optional[str] = None
    role: Optional[str] = None
    assigned_area: Optional[str] = None
    assigned_waste_type: Optional[str] = None
    password: Optional[str] = None   # if provided, updates Firebase Auth password

class SkipStopRequest(BaseModel):
    route_id:       str
    bin_id:         str
    worker_id:      str
    exception_type: str   # "bin_inaccessible" | "bin_damaged" | "hazardous_material"
    exception_note: str = ""

class ClockInRequest(BaseModel):
    worker_id: str

class ClockOutRequest(BaseModel):
    worker_id: str

class ResolveExceptionRequest(BaseModel):
    status:      str             # "resolved" | "escalated"
    resolved_by: str = ""
    notes:       Optional[str] = ""

class ScheduleShiftRequest(BaseModel):
    worker_id:      str
    scheduled_date: str          # YYYY-MM-DD
    start_time:     str          # HH:MM (24h)
    end_time:       str          # HH:MM (24h)
    note:           str = ""
    created_by:     str = "admin"

# ── Settings helper ───────────────────────────────────────────────────────────
def _get_thresholds() -> tuple[int, int]:
    """Read warning and critical fill thresholds from Firestore settings/main.
    Admin Panel saves these via the Settings → General → Collection Settings sliders.
    Falls back to (70, 90) if the document doesn't exist or a read error occurs."""
    try:
        snap = db.collection("settings").document("main").get()
        if snap.exists:
            data     = snap.to_dict() or {}
            warning  = int(data.get("warningThreshold",  70))
            critical = int(data.get("criticalThreshold", 90))
            # Basic sanity: warning must be strictly less than critical
            if 0 < warning < critical <= 100:
                return warning, critical
    except Exception as e:
        print(f"[settings] Could not read thresholds, using defaults: {e}")
    return 70, 90

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

def _two_opt_improve(stops: list, depot_lat: float, depot_lng: float) -> list:
    """2-opt local-search refinement after greedy ordering. Caps at 3 passes."""
    best = list(stops)
    best_dist = _total_route_km(best, depot_lat, depot_lng)
    improved = True
    passes = 0
    while improved and passes < 3:
        improved = False
        passes += 1
        for i in range(len(best) - 1):
            for j in range(i + 2, len(best)):
                candidate = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                dist = _total_route_km(candidate, depot_lat, depot_lng)
                if dist < best_dist - 0.001:
                    best, best_dist, improved = candidate, dist, True
    return best

def _total_route_km(ordered: list, depot_lat: float, depot_lng: float) -> float:
    total = 0.0
    prev_lat, prev_lng = depot_lat, depot_lng
    for stop in ordered:
        total += _haversine_km(prev_lat, prev_lng, stop["lat"], stop["lng"])
        prev_lat, prev_lng = stop["lat"], stop["lng"]
    total += _haversine_km(prev_lat, prev_lng, depot_lat, depot_lng)
    return round(total, 3)

def _is_quiet_hours() -> bool:
    """True when Islamabad local time (PKT, UTC+5) is between 23:00 and 05:00 — no heavy-waste collection at night."""
    hour = datetime.now(ZoneInfo("Asia/Karachi")).hour
    return hour >= 23 or hour < 5

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
    lat:    Optional[float] = Query(default=None),
    lng:    Optional[float] = Query(default=None),
    area:   Optional[str]   = Query(default=None),
    bin_id: Optional[str]   = Query(default=None),
):
    # Bins are ONLY created through this endpoint — never seed them manually.
    print("\n========== STARTING AI ANALYSIS ==========")
    try:
        print("Step 1: Reading uploaded image...")
        image_bytes = await image_file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        print(f"-> Image read OK ({len(image_bytes)} bytes, size={image.size})")

        print("Step 2: Running Fill Level Model...")
        fill_status = "Not Detected"
        fill_confidence = 0.0
        fill_results = _models["fill"](image, conf=0.5)
        if len(fill_results[0].boxes) > 0:
            best = fill_results[0].boxes[0]
            raw_label = _models["fill"].names[int(best.cls[0])]
            fill_status = raw_label.strip().title()
            fill_confidence = round(float(best.conf[0]), 3)
        fill_level_int = _label_to_fill_int(fill_status)
        print(f"-> Fill Model OK: label='{fill_status}', level={fill_level_int}%, conf={fill_confidence}")

        # ── Abort early: model found no bin in the image ──────────────────────
        if fill_status == "Not Detected":
            print("-> Detection failed: no bin found. Aborting without Firestore write.")
            raise HTTPException(
                status_code=422,
                detail="no_bin_detected",
            )

        detected_waste = []
        primary_waste_type = "Unknown"

        if fill_status not in ("Empty",):
            print("Step 3: Running Waste Type Model...")
            waste_results = _models["waste"](image, conf=0.5)
            for box in waste_results[0].boxes:
                detected_waste.append({
                    "type":       _models["waste"].names[int(box.cls[0])],
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

        # Step 3b: Build compressed thumbnail for Firestore image preview
        print("Step 3b: Generating image thumbnail...")
        thumb = image.copy()
        thumb.thumbnail((640, 640))
        thumb_buf = io.BytesIO()
        thumb.save(thumb_buf, format="JPEG", quality=60)
        image_preview_b64 = base64.b64encode(thumb_buf.getvalue()).decode("utf-8")
        print(f"-> Thumbnail OK ({len(thumb_buf.getvalue())} bytes → {len(image_preview_b64)} chars base64)")

        print("Step 4: Writing to Firestore...")
        warning_thresh, critical_thresh = _get_thresholds()
        bin_status = (
            "critical" if fill_level_int >= critical_thresh
            else "warning" if fill_level_int >= warning_thresh
            else "normal"
        )
        bin_lat  = lat if lat is not None else 0.0
        bin_lng  = lng if lng is not None else 0.0
        bin_area = (area or "").strip()
        bin_data: dict = {
            "fillStatus":    fill_status,
            "fillLevel":     fill_level_int,
            "wasteType":     primary_waste_type,
            "aiConfidence":  fill_confidence,
            "detectedWaste": detected_waste,
            "lat":           bin_lat,
            "lng":           bin_lng,
            "isLocked":      False,
            "status":        bin_status,
            "capacity":      240,
            "area":          bin_area,
            "sector":        bin_area,
            "lastAnalyzed":  datetime.now(timezone.utc),
            "imagePreview":  image_preview_b64,
        }

        # Update existing bin if bin_id given, otherwise create/update by lat+lng
        if bin_id:
            bin_ref = db.collection("bins").document(bin_id)
            if bin_ref.get().exists:
                # Preserve isLocked / routeId — don't overwrite routing state
                update_fields = {k: v for k, v in bin_data.items() if k not in ("isLocked",)}
                bin_ref.update(update_fields)
                saved_bin_id = bin_id
                print(f"-> Updated existing bin: {saved_bin_id}")
            else:
                bin_ref.set(bin_data)
                saved_bin_id = bin_id
                print(f"-> Created bin with provided id: {saved_bin_id}")
        else:
            # No bin_id: use a deterministic document ID derived from the
            # rounded lat/lng so the same coordinates always map to the same
            # Firestore document — no query needed, no composite index required.
            def _coord_seg(v: float) -> str:
                return f"{v:.5f}".replace(".", "d").replace("-", "m")

            bin_doc_id = f"bin_{_coord_seg(bin_lat)}_{_coord_seg(bin_lng)}"
            bin_ref = db.collection("bins").document(bin_doc_id)
            existing = bin_ref.get()
            if existing.exists:
                update_fields = {k: v for k, v in bin_data.items() if k not in ("isLocked",)}
                bin_ref.update(update_fields)
                saved_bin_id = bin_doc_id
                print(f"-> Updated existing bin (same lat/lng): {saved_bin_id}")
            else:
                bin_ref.set(bin_data)
                saved_bin_id = bin_doc_id
                print(f"-> Created new bin: {saved_bin_id}")

        # --- SFR-017: FCM Push Notifications for Urgent Overflows ---
        if fill_level_int >= critical_thresh:
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
                    db.collection("bins").document(saved_bin_id).update({"isLocked": True, "routeId": route_doc.id, "area": bin_area})

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
                    route_driver_id = r_data.get("driverId")
                    if route_driver_id:
                        worker_snap = db.collection("users").document(route_driver_id).get()
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
            "success":      True,
            "bin_id":       saved_bin_id,
            "fillStatus":   fill_status,
            "fillLevel":    fill_level_int,
            "wasteType":    primary_waste_type,
            "aiConfidence": fill_confidence,
            "status":       bin_status,
            "capacity":     240,
            "area":         bin_area,
            "lat":          bin_lat,
            "lng":          bin_lng,
            # Nested results kept for backward compatibility with client scan display
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


# ── 1b. Re-scan an existing bin (no lat/lng needed) ──────────────────────────
# POST /rescan-bin/{bin_id}  +  image_file (multipart)
# Runs the same AI pipeline as /analyze/ but always updates the existing
# document — lat, lng, area, isLocked, and routeId are never touched.
@app.post("/rescan-bin/{bin_id}")
async def rescan_bin(bin_id: str, image_file: UploadFile = File(...)):
    print(f"\n========== RESCAN: {bin_id} ==========")
    try:
        # 1. Verify the bin exists and grab its preserved fields
        bin_ref  = db.collection("bins").document(bin_id)
        bin_snap = bin_ref.get()
        if not bin_snap.exists:
            raise HTTPException(status_code=404, detail=f"Bin '{bin_id}' not found.")
        existing = bin_snap.to_dict()

        # 2. Read + decode image
        image_bytes = await image_file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        print(f"-> Image OK ({len(image_bytes)} bytes, size={image.size})")

        # 3. Fill level model
        fill_status    = "Not Detected"
        fill_confidence = 0.0
        fill_results   = _models["fill"](image, conf=0.5)
        if len(fill_results[0].boxes) > 0:
            best           = fill_results[0].boxes[0]
            fill_status    = _models["fill"].names[int(best.cls[0])].strip().title()
            fill_confidence = round(float(best.conf[0]), 3)
        fill_level_int = _label_to_fill_int(fill_status)
        print(f"-> Fill: '{fill_status}' ({fill_level_int}%, conf={fill_confidence})")

        if fill_status == "Not Detected":
            raise HTTPException(status_code=422, detail="no_bin_detected")

        # 4. Waste type model (skip for Empty bins)
        primary_waste_type = existing.get("wasteType", "Unknown")
        detected_waste     = []
        if fill_status not in ("Empty",):
            waste_results = _models["waste"](image, conf=0.5)
            for box in waste_results[0].boxes:
                detected_waste.append({
                    "type":       _models["waste"].names[int(box.cls[0])],
                    "confidence": round(float(box.conf[0]), 3),
                })
            if detected_waste:
                primary_waste_type = max(detected_waste, key=lambda w: w["confidence"])["type"]
            print(f"-> Waste: '{primary_waste_type}'")

        # 5. Thumbnail
        thumb = image.copy()
        thumb.thumbnail((640, 640))
        buf = io.BytesIO()
        thumb.save(buf, format="JPEG", quality=60)
        image_preview_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        # 6. Status
        warning_thresh, critical_thresh = _get_thresholds()
        bin_status = (
            "critical" if fill_level_int >= critical_thresh
            else "warning" if fill_level_int >= warning_thresh
            else "normal"
        )

        # 7. Update only AI-derived fields — preserve lat/lng/area/isLocked/routeId
        bin_ref.update({
            "fillStatus":    fill_status,
            "fillLevel":     fill_level_int,
            "wasteType":     primary_waste_type,
            "aiConfidence":  fill_confidence,
            "detectedWaste": detected_waste,
            "imagePreview":  image_preview_b64,
            "status":        bin_status,
            "lastAnalyzed":  datetime.now(timezone.utc),
        })
        print(f"-> Updated bin {bin_id}")
        print("========== RESCAN COMPLETE ==========\n")
        return {
            "success":      True,
            "bin_id":       bin_id,
            "fillStatus":   fill_status,
            "fillLevel":    fill_level_int,
            "wasteType":    primary_waste_type,
            "aiConfidence": fill_confidence,
            "status":       bin_status,
        }

    except HTTPException:
        raise
    except Exception as e:
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
    quiet_hours    = _is_quiet_hours()

    # Reject GPS depot at (0,0) — this means location services are off on the device
    if req.depot_lat == 0.0 and req.depot_lng == 0.0:
        raise HTTPException(
            status_code=422,
            detail="Worker GPS is unavailable (coordinates 0,0 are invalid). Enable location services and try again."
        )

    # 2. Read admin-configurable thresholds, then fetch qualifying bins
    collection_threshold, critical_threshold = _get_thresholds()
    bin_query = db.collection("bins").where(
        filter=FieldFilter("fillLevel", ">=", collection_threshold)
    ).stream()
    available_stops = []

    for doc in bin_query:
        data = doc.to_dict()

        # Concurrency: skip bins already locked by another active route
        if data.get("isLocked", False):
            continue

        # Safety: skip bins flagged as hazardous by a driver exception
        if data.get("hasHazardousFlag", False):
            continue

        lat = float(data.get("lat", 0))
        lng = float(data.get("lng", 0))
        if lat == 0.0 and lng == 0.0:
            continue

        # Area filter: assigned workers only see their district.
        # Unassigned workers (no assignedArea) fall back to a 10 km proximity
        # filter centred on their depot GPS so the route matches what is shown
        # on the map.
        bin_area = data.get("area", data.get("sector", ""))
        if assigned_area:
            # Case-insensitive hierarchy match: exact OR bin area ends with ", <assigned_area>".
            # Lets a city-level worker ("Lahore") see sub-area bins
            # ("Wahdat Colony, Lahore", "DHA, Lahore") without widening to
            # completely unrelated areas.
            _ba = bin_area.strip().lower()
            _aa = assigned_area.strip().lower()
            if not (_ba == _aa or _ba.endswith(f", {_aa}")):
                continue
        else:
            dist_km = _haversine_km(req.depot_lat, req.depot_lng, lat, lng)
            if dist_km > 10:
                continue

        # Waste type filter: worker only handles their assigned stream.
        # "Mixed"/"All"/"Any"/"General" are wildcards meaning "collect everything".
        waste_type = data.get("wasteType", "")
        WILDCARD_WASTE = {"mixed", "all", "any", "general"}
        if assigned_waste and assigned_waste not in WILDCARD_WASTE \
                and waste_type.lower() != assigned_waste:
            continue

        # German noise regulation: no Glass or Metal collections at night
        if quiet_hours and waste_type.lower() in NOISY_WASTE_TYPES:
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
        detail = f"No bins ≥ {collection_threshold}% full found in your area. Nothing to collect right now."
        if quiet_hours:
            detail += " Note: Glass/Metal bins are excluded during quiet hours (23:00–05:00 Islamabad time)."
        raise HTTPException(status_code=404, detail=detail)

    # 3. Greedy Nearest-Neighbor ordering, then 2-opt refinement
    ordered_stops = _greedy_nearest_neighbor(available_stops, req.depot_lat, req.depot_lng)
    if len(ordered_stops) > 2:
        ordered_stops = _two_opt_improve(ordered_stops, req.depot_lat, req.depot_lng)

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
        .where(filter=FieldFilter("driverId", "==", worker_id))
        .where(filter=FieldFilter("status", "==", "active"))
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
    found = False
    for stop in stops:
        if stop.get("binId") == req.bin_id:
            if stop.get("completed", False):
                raise HTTPException(
                    status_code=409,
                    detail=f"Stop '{req.bin_id}' is already marked as completed."
                )
            stop["completed"]  = True
            fill_at_pickup     = stop.get("fillLevel", 0)
            bin_area           = stop.get("area", "")
            bin_waste_type     = stop.get("wasteType", "")
            bin_lat            = float(stop.get("lat", 0))
            bin_lng            = float(stop.get("lng", 0))
            found = True
            break

    if not found:
        raise HTTPException(
            status_code=404,
            detail=f"Bin '{req.bin_id}' is not a stop on route '{req.route_id}'."
        )

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

    # Unlock the bin and reset fill level (bin may have been deleted by admin — skip gracefully)
    bin_doc_ref = db.collection("bins").document(req.bin_id)
    if bin_doc_ref.get().exists:
        bin_doc_ref.update({
            "fillLevel":     0,
            "fillStatus":    "Empty",
            "status":        "normal",
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

    # Increment worker's lifetime collection counter — drives WorkerPerformanceChart
    if req.worker_id:
        try:
            db.collection("users").document(req.worker_id).update({
                "collections": firestore.Increment(1)
            })
        except Exception:
            pass  # non-critical: chart will self-correct on next collection

    return {
        "success":         True,
        "route_status":    new_status,
        "completed_stops": new_completed,
        "total_stops":     total_stops,
    }


# ── Admin: pickup logs ────────────────────────────────────────────────────────
@app.get("/admin/pickup-logs")
def admin_pickup_logs(worker_id: Optional[str] = None, limit: int = 20, page: int = 1):
    limit  = min(max(limit, 1), 100)
    page   = max(page, 1)
    offset = (page - 1) * limit

    base_q = db.collection("pickupLogs").order_by("completedAt", direction=firestore.Query.DESCENDING)
    if worker_id:
        base_q = base_q.where(filter=FieldFilter("workerId", "==", worker_id))

    # Fetch one extra doc to cheaply detect whether a next page exists
    fetched = list(base_q.limit(offset + limit + 1).stream())
    has_more = len(fetched) > offset + limit
    page_docs = fetched[offset : offset + limit]
    logs = [doc.to_dict() for doc in page_docs]
    return {"success": True, "logs": logs, "count": len(logs), "page": page, "has_more": has_more}


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
    # One worker per area constraint
    if req.role == "worker" and req.assigned_area.strip():
        area_norm = req.assigned_area.strip().lower()
        existing_workers = db.collection("users").where(
            filter=FieldFilter("role", "==", "worker")
        ).stream()
        for w in existing_workers:
            w_data = w.to_dict()
            if w_data.get("assignedArea", "").strip().lower() == area_norm:
                w_name = (
                    f"{w_data.get('firstName', '')} {w_data.get('lastName', '')}".strip()
                    or w_data.get("email", "another worker")
                )
                raise HTTPException(
                    status_code=409,
                    detail=f"Area '{req.assigned_area.strip()}' is already assigned to {w_name}. Each area can only have one worker.",
                )

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


# ── Admin: Update user (Firestore + optionally Firebase Auth password) ────────
@app.patch("/admin/update-user/{uid}")
def admin_update_user(uid: str, req: UpdateUserRequest):
    user_doc = db.collection("users").document(uid).get()
    if not user_doc.exists:
        raise HTTPException(status_code=404, detail=f"User '{uid}' not found.")

    current = user_doc.to_dict()
    new_role = req.role if req.role is not None else current.get("role", "worker")

    # One-worker-per-area constraint (skip current user)
    if new_role == "worker" and req.assigned_area is not None and req.assigned_area.strip():
        area_norm = req.assigned_area.strip().lower()
        for w in db.collection("users").where(filter=FieldFilter("role", "==", "worker")).stream():
            if w.id == uid:
                continue
            w_data = w.to_dict()
            if w_data.get("assignedArea", "").strip().lower() == area_norm:
                w_name = (
                    f"{w_data.get('firstName', '')} {w_data.get('lastName', '')}".strip()
                    or w_data.get("email", "another worker")
                )
                raise HTTPException(
                    status_code=409,
                    detail=f"Area '{req.assigned_area.strip()}' is already assigned to {w_name}.",
                )

    updates: dict = {}
    if req.first_name is not None:
        updates["firstName"] = req.first_name.strip()
    if req.last_name is not None:
        updates["lastName"] = req.last_name.strip()
    if req.phone is not None:
        updates["phoneNumber"] = req.phone.strip()
    if req.role is not None:
        updates["role"] = req.role
        if req.role != "worker":
            updates["assignedArea"] = ""
            updates["assignedWasteType"] = ""
    if req.assigned_area is not None and new_role == "worker":
        updates["assignedArea"] = req.assigned_area.strip()
    if req.assigned_waste_type is not None and new_role == "worker":
        updates["assignedWasteType"] = req.assigned_waste_type

    if updates:
        db.collection("users").document(uid).update(updates)

    if req.password and len(req.password) >= 6:
        try:
            fb_auth.update_user(uid, password=req.password)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Password update failed: {e}")

    return {"success": True, "uid": uid}


# ── Admin: Delete bin + remove from active routes ─────────────────────────────
@app.delete("/admin/delete-bin/{bin_id}")
def admin_delete_bin(bin_id: str):
    """Deletes a bin document and scrubs it from every active/pending route."""
    # 1. Remove bin from any routes that reference it
    affected_routes = list(
        db.collection("routes")
        .where(filter=FieldFilter("status", "in", ["active", "pending"]))
        .stream()
    )
    for route_doc in affected_routes:
        r_data = route_doc.to_dict()
        stops = r_data.get("stops", [])
        new_stops = [s for s in stops if s.get("binId") != bin_id]
        if len(new_stops) != len(stops):
            new_total = len(new_stops)
            completed = sum(1 for s in new_stops if s.get("completed", False))
            new_status = "completed" if (new_total > 0 and completed >= new_total) else r_data.get("status", "active")
            route_doc.reference.update({
                "stops":          new_stops,
                "totalStops":     new_total,
                "status":         new_status,
            })

    # 2. Delete the bin document itself
    db.collection("bins").document(bin_id).delete()
    return {"success": True, "bin_id": bin_id}


# ── Admin KPI Dashboard ───────────────────────────────────────────────────────
@app.get("/admin/stats")
def admin_stats(days: int = 30):
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    stats_warning_thresh, _ = _get_thresholds()
    full_bins = list(
        db.collection("bins")
        .where(filter=FieldFilter("fillLevel", ">=", stats_warning_thresh))
        .stream()
    )
    active_workers = list(
        db.collection("users")
        .where(filter=FieldFilter("role", "==", "worker"))
        .where(filter=FieldFilter("status", "==", "active"))
        .stream()
    )
    # Only scan routes within the window to avoid unbounded reads
    all_routes = [
        doc.to_dict()
        for doc in db.collection("routes")
        .where(filter=FieldFilter("createdAt", ">=", cutoff))
        .stream()
    ]
    active_routes    = [r for r in all_routes if r.get("status") == "active"]
    completed_routes = [r for r in all_routes if r.get("status") == "completed"]

    total_fuel   = round(sum(r.get("estimatedFuel", 0)     for r in completed_routes), 2)
    total_carbon = round(sum(r.get("carbonFootprintKg", 0) for r in completed_routes), 3)

    total_stops = sum(r.get("totalStops", 0)     for r in completed_routes)
    done_stops  = sum(r.get("completedStops", 0) for r in completed_routes)
    efficiency  = round(done_stops / total_stops * 100, 1) if total_stops > 0 else 0.0

    pending_anomalies = list(db.collection("anomalies").where(filter=FieldFilter("status", "==", "pending")).stream())
    clocked_in_users  = list(db.collection("users").where(filter=FieldFilter("shiftStatus", "==", "clocked_in")).stream())
    open_exceptions   = list(db.collection("routeExceptions").where(filter=FieldFilter("status", "==", "open")).stream())

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
            "activeShifts":        len(clocked_in_users),
            "openExceptions":      len(open_exceptions),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 3 — EXCEPTION HANDLING, SHIFT MANAGEMENT, MODEL HOT-SWAP
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/skip-stop")
def skip_stop(req: SkipStopRequest):
    now_utc   = datetime.now(timezone.utc)
    route_ref = db.collection("routes").document(req.route_id)
    route_snap = route_ref.get()
    if not route_snap.exists:
        raise HTTPException(status_code=404, detail=f"Route '{req.route_id}' not found.")

    route_data = route_snap.to_dict()
    stops = list(route_data.get("stops", []))

    bin_area, bin_lat, bin_lng = "", 0.0, 0.0
    found = False
    for stop in stops:
        if stop.get("binId") == req.bin_id:
            if stop.get("skipped", False):
                raise HTTPException(status_code=409, detail="already_skipped")
            if stop.get("completed", False):
                raise HTTPException(status_code=409, detail="already_completed")
            stop["skipped"]       = True
            stop["exceptionType"] = req.exception_type
            stop["exceptionNote"] = req.exception_note
            stop["exceptionAt"]   = now_utc
            bin_area = stop.get("area", "")
            bin_lat  = float(stop.get("lat", 0))
            bin_lng  = float(stop.get("lng", 0))
            found = True
            break

    if not found:
        raise HTTPException(status_code=404, detail=f"Bin '{req.bin_id}' not found in route stops.")

    skipped_stops  = route_data.get("skippedStops", 0) + 1
    completed_done = route_data.get("completedStops", 0)
    total_stops    = route_data.get("totalStops", 0)
    new_status     = "completed" if (completed_done + skipped_stops) >= total_stops else "active"

    update: dict = {"stops": stops, "skippedStops": skipped_stops, "status": new_status}
    if new_status == "completed":
        update["completedAt"] = now_utc

    # Detect route abandonment: route completed with zero collections (all stops skipped)
    is_abandoned = (
        new_status == "completed"
        and completed_done == 0
        and skipped_stops == total_stops
        and total_stops > 0
    )
    if is_abandoned:
        update["abandoned"] = True

    route_ref.update(update)

    # Write abandonment alert to Firestore so Dashboard can surface it immediately
    if is_abandoned:
        # worker_name is resolved later for the exception doc — read it now
        _ws = db.collection("users").document(req.worker_id).get()
        _wn = ""
        if _ws.exists:
            _wd = _ws.to_dict()
            _wn = f"{_wd.get('firstName', '')} {_wd.get('lastName', '')}".strip()
        alert_id = str(uuid.uuid4())
        db.collection("alerts").document(alert_id).set({
            "alertId":    alert_id,
            "type":       "route_abandoned",
            "routeId":    req.route_id,
            "workerId":   req.worker_id,
            "workerName": _wn or req.worker_id,
            "totalStops": total_stops,
            "area":       route_data.get("assignedArea", ""),
            "status":     "unread",
            "createdAt":  now_utc,
        })

    # Unlock bin (keep fillLevel — not yet collected)
    bin_ref = db.collection("bins").document(req.bin_id)
    if bin_ref.get().exists:
        bin_update: dict = {"isLocked": False, "routeId": firestore.DELETE_FIELD}
        if req.exception_type == "hazardous_material":
            bin_update["hasHazardousFlag"]     = True
            bin_update["hazardousFlaggedAt"]   = now_utc
        bin_ref.update(bin_update)

    # Resolve worker name for the exception doc
    worker_snap = db.collection("users").document(req.worker_id).get()
    worker_data = worker_snap.to_dict() if worker_snap.exists else {}
    worker_name = f"{worker_data.get('firstName', '')} {worker_data.get('lastName', '')}".strip() or req.worker_id

    exception_id = str(uuid.uuid4())
    db.collection("routeExceptions").document(exception_id).set({
        "exceptionId":   exception_id,
        "routeId":       req.route_id,
        "binId":         req.bin_id,
        "workerId":      req.worker_id,
        "workerName":    worker_name,
        "exceptionType": req.exception_type,
        "exceptionNote": req.exception_note,
        "area":          bin_area,
        "binLat":        bin_lat,
        "binLng":        bin_lng,
        "status":        "open",
        "resolvedBy":    None,
        "resolvedAt":    None,
        "createdAt":     now_utc,
    })

    return {
        "success":         True,
        "exception_id":    exception_id,
        "route_status":    new_status,
        "completed_stops": completed_done,
        "skipped_stops":   skipped_stops,
        "total_stops":     total_stops,
    }


@app.post("/clock-in")
def clock_in(req: ClockInRequest):
    user_ref  = db.collection("users").document(req.worker_id)
    user_snap = user_ref.get()
    if not user_snap.exists:
        raise HTTPException(status_code=404, detail=f"Worker '{req.worker_id}' not found.")

    user_data = user_snap.to_dict()
    if user_data.get("shiftStatus") == "clocked_in":
        raise HTTPException(status_code=409, detail="already_clocked_in")

    # If a scheduled shift exists for today, block clock-in before start time.
    # Workers may clock in up to 5 minutes early; no restriction if no schedule exists.
    now_pkt   = datetime.now(ZoneInfo("Asia/Karachi"))
    today_str = now_pkt.strftime("%Y-%m-%d")
    scheduled_today = list(
        db.collection("shiftSchedules")
        .where(filter=FieldFilter("workerId", "==", req.worker_id))
        .where(filter=FieldFilter("scheduledDate", "==", today_str))
        .where(filter=FieldFilter("status", "in", ["scheduled", "acknowledged"]))
        .limit(1)
        .stream()
    )
    if not scheduled_today:
        raise HTTPException(status_code=409, detail="no_shift_scheduled")

    start_str = scheduled_today[0].to_dict().get("startTime", "00:00")
    try:
        h, m = map(int, start_str.split(":"))
        shift_start = now_pkt.replace(hour=h, minute=m, second=0, microsecond=0)
        if now_pkt < shift_start - timedelta(minutes=5):
            raise HTTPException(status_code=409, detail="shift_not_started")
    except (ValueError, AttributeError):
        pass  # unparseable time — don't block

    worker_name = f"{user_data.get('firstName', '')} {user_data.get('lastName', '')}".strip()
    now_utc     = datetime.now(timezone.utc)
    shift_id    = str(uuid.uuid4())

    db.collection("shifts").document(shift_id).set({
        "shiftId":            shift_id,
        "workerId":           req.worker_id,
        "workerName":         worker_name,
        "assignedArea":       user_data.get("assignedArea", ""),
        "assignedWasteType":  user_data.get("assignedWasteType", ""),
        "clockInTime":        now_utc,
        "clockOutTime":       None,
        "totalDurationMinutes": None,
        "routesCompleted":    0,
        "collectionsCompleted": 0,
        "status":             "active",
    })

    user_ref.update({
        "shiftStatus":    "clocked_in",
        "currentShiftId": shift_id,
        "lastClockIn":    now_utc,
    })

    # Mark today's scheduled shift as acknowledged
    today_str = now_utc.strftime("%Y-%m-%d")
    scheduled = list(
        db.collection("shiftSchedules")
        .where(filter=FieldFilter("workerId", "==", req.worker_id))
        .where(filter=FieldFilter("scheduledDate", "==", today_str))
        .where(filter=FieldFilter("status", "==", "scheduled"))
        .limit(1)
        .stream()
    )
    if scheduled:
        scheduled[0].reference.update({"status": "acknowledged"})

    return {"success": True, "shift_id": shift_id, "clock_in_time": now_utc.isoformat()}


@app.post("/clock-out")
def clock_out(req: ClockOutRequest):
    user_ref  = db.collection("users").document(req.worker_id)
    user_snap = user_ref.get()
    if not user_snap.exists:
        raise HTTPException(status_code=404, detail=f"Worker '{req.worker_id}' not found.")

    user_data = user_snap.to_dict()
    shift_id  = user_data.get("currentShiftId")
    if not shift_id:
        raise HTTPException(status_code=409, detail="not_clocked_in")

    shift_ref  = db.collection("shifts").document(shift_id)
    shift_snap = shift_ref.get()
    if not shift_snap.exists:
        raise HTTPException(status_code=404, detail=f"Shift '{shift_id}' not found.")

    shift_data = shift_snap.to_dict()
    clock_in_ts = shift_data.get("clockInTime")
    now_utc     = datetime.now(timezone.utc)

    duration_mins = 0
    if clock_in_ts:
        if hasattr(clock_in_ts, "replace"):
            clock_in_dt = clock_in_ts.replace(tzinfo=timezone.utc) if clock_in_ts.tzinfo is None else clock_in_ts
        else:
            clock_in_dt = clock_in_ts
        duration_mins = int((now_utc - clock_in_dt).total_seconds() / 60)

    # Count routes completed since clock-in
    routes_done = list(
        db.collection("routes")
        .where(filter=FieldFilter("driverId", "==", req.worker_id))
        .where(filter=FieldFilter("status", "==", "completed"))
        .stream()
    )
    if clock_in_ts:
        routes_done = [r for r in routes_done if r.to_dict().get("createdAt", now_utc) >= clock_in_ts]
    routes_count = len(routes_done)

    # Count pickups since clock-in
    logs_done = list(
        db.collection("pickupLogs")
        .where(filter=FieldFilter("workerId", "==", req.worker_id))
        .stream()
    )
    if clock_in_ts:
        logs_done = [l for l in logs_done if l.to_dict().get("completedAt", now_utc) >= clock_in_ts]
    collections_count = len(logs_done)

    shift_ref.update({
        "clockOutTime":          now_utc,
        "totalDurationMinutes":  duration_mins,
        "routesCompleted":       routes_count,
        "collectionsCompleted":  collections_count,
        "status":                "completed",
    })

    user_ref.update({
        "shiftStatus":    "clocked_out",
        "currentShiftId": None,
        "lastClockOut":   now_utc,
    })

    # Mark today's scheduled shift as completed
    today_str = now_utc.strftime("%Y-%m-%d")
    scheduled = list(
        db.collection("shiftSchedules")
        .where(filter=FieldFilter("workerId", "==", req.worker_id))
        .where(filter=FieldFilter("scheduledDate", "==", today_str))
        .where(filter=FieldFilter("status", "in", ["scheduled", "acknowledged"]))
        .limit(1)
        .stream()
    )
    if scheduled:
        scheduled[0].reference.update({"status": "completed"})

    return {
        "success":              True,
        "shift_id":             shift_id,
        "duration_minutes":     duration_mins,
        "routes_completed":     routes_count,
        "collections_completed": collections_count,
    }


@app.get("/worker/shift-status/{worker_id}")
def worker_shift_status(worker_id: str):
    snap = db.collection("users").document(worker_id).get()
    if not snap.exists:
        raise HTTPException(status_code=404, detail=f"Worker '{worker_id}' not found.")
    data = snap.to_dict()
    clock_in_ts = data.get("lastClockIn")
    return {
        "is_clocked_in": data.get("shiftStatus") == "clocked_in",
        "shift_id":      data.get("currentShiftId"),
        "clock_in_time": clock_in_ts.isoformat() if clock_in_ts and hasattr(clock_in_ts, "isoformat") else None,
    }


@app.get("/admin/shifts")
def admin_shifts(date: Optional[str] = None, worker_id: Optional[str] = None,
                 status: Optional[str] = None, limit: int = 50):
    limit = min(limit, 200)
    q = db.collection("shifts").order_by("clockInTime", direction=firestore.Query.DESCENDING)

    if status:
        q = db.collection("shifts").where(filter=FieldFilter("status", "==", status)).order_by("clockInTime", direction=firestore.Query.DESCENDING)

    if worker_id:
        q = db.collection("shifts").where(filter=FieldFilter("workerId", "==", worker_id)).order_by("clockInTime", direction=firestore.Query.DESCENDING)

    docs = [d.to_dict() for d in q.limit(limit).stream()]

    if date:
        try:
            day_start = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            day_end   = day_start + timedelta(days=1)
            docs = [
                d for d in docs
                if d.get("clockInTime") and day_start <= d["clockInTime"].replace(tzinfo=timezone.utc) < day_end
            ]
        except ValueError:
            pass

    return {"success": True, "shifts": docs, "count": len(docs)}


@app.get("/admin/exceptions")
def admin_exceptions(status: Optional[str] = "open", worker_id: Optional[str] = None,
                     area: Optional[str] = None, exception_type: Optional[str] = None,
                     limit: int = 50):
    limit = min(limit, 200)
    if status:
        q = (db.collection("routeExceptions")
             .where(filter=FieldFilter("status", "==", status))
             .order_by("createdAt", direction=firestore.Query.DESCENDING)
             .limit(limit))
    else:
        q = (db.collection("routeExceptions")
             .order_by("createdAt", direction=firestore.Query.DESCENDING)
             .limit(limit))

    docs = [d.to_dict() for d in q.stream()]

    if worker_id:
        docs = [d for d in docs if d.get("workerId") == worker_id]
    if area:
        docs = [d for d in docs if area.lower() in d.get("area", "").lower()]
    if exception_type:
        docs = [d for d in docs if d.get("exceptionType") == exception_type]

    return {"success": True, "exceptions": docs, "count": len(docs)}


@app.patch("/admin/exceptions/{exception_id}")
def resolve_exception(exception_id: str, req: ResolveExceptionRequest):
    ref  = db.collection("routeExceptions").document(exception_id)
    snap = ref.get()
    if not snap.exists:
        raise HTTPException(status_code=404, detail=f"Exception '{exception_id}' not found.")

    update: dict = {"status": req.status}
    if req.status in ("resolved", "escalated"):
        update["resolvedBy"] = req.resolved_by
        update["resolvedAt"] = datetime.now(timezone.utc)
    if req.notes:
        update["notes"] = req.notes
    ref.update(update)
    return {"success": True, "exception_id": exception_id, "new_status": req.status}


@app.post("/admin/upload-model")
async def upload_model(
    model_file:  UploadFile = File(...),
    model_type:  str        = Form(...),   # "fill" | "waste"
    version_tag: str        = Form(...),
):
    if model_type not in ("fill", "waste"):
        raise HTTPException(status_code=400, detail="model_type must be 'fill' or 'waste'.")
    if not model_file.filename.endswith(".pt"):
        raise HTTPException(status_code=400, detail="Only .pt files are accepted.")

    models_dir  = Path("models")
    pending_path = models_dir / f"{model_type}_model_pending.pt"
    final_path   = models_dir / f"{model_type}_model.pt"
    backup_path  = models_dir / f"{model_type}_model_backup.pt"

    # Save upload to staging path
    content = await model_file.read()
    pending_path.write_bytes(content)

    # Validate with a single test inference
    try:
        _orig = torch.load
        torch.load = lambda *a, **kw: _orig(*a, **{**kw, "weights_only": False})
        test_model = YOLO(str(pending_path))
        torch.load = _orig

        test_img_path = next(Path("asset/bins").glob("*.jpg"), None) if Path("asset/bins").exists() else None
        conf_score = 0.0
        if test_img_path:
            test_img = Image.open(test_img_path).convert("RGB")
            result   = test_model(test_img, conf=0.1)
            if result and len(result[0].boxes) > 0:
                conf_score = round(float(result[0].boxes[0].conf[0]), 3)
    except Exception as e:
        pending_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=f"Model validation failed: {e}")

    # Promote to active: keep old as backup
    if final_path.exists():
        final_path.replace(backup_path)
    pending_path.replace(final_path)

    # Reload models into memory
    _load_models()

    # Update Firestore settings
    field = "fillModelVersion" if model_type == "fill" else "wasteModelVersion"
    db.collection("settings").document("main").set(
        {field: version_tag, "lastModelUpdate": datetime.now(timezone.utc)},
        merge=True,
    )

    return {"success": True, "model_type": model_type, "version": version_tag, "test_confidence": conf_score}


@app.post("/admin/reload-models")
def reload_models_endpoint():
    _load_models()
    db.collection("settings").document("main").set(
        {"lastModelUpdate": datetime.now(timezone.utc)}, merge=True
    )
    return {"success": True, "models_loaded": list(_models.keys())}


# ── Shift Scheduling ──────────────────────────────────────────────────────────

@app.post("/admin/schedule-shift")
def admin_schedule_shift(req: ScheduleShiftRequest):
    worker_snap = db.collection("users").document(req.worker_id).get()
    if not worker_snap.exists:
        raise HTTPException(status_code=404, detail=f"Worker '{req.worker_id}' not found.")

    worker = worker_snap.to_dict()
    worker_name = f"{worker.get('firstName', '')} {worker.get('lastName', '')}".strip()

    schedule_id = str(uuid.uuid4())
    db.collection("shiftSchedules").document(schedule_id).set({
        "scheduleId":       schedule_id,
        "workerId":         req.worker_id,
        "workerName":       worker_name,
        "assignedArea":     worker.get("assignedArea", ""),
        "assignedWasteType":worker.get("assignedWasteType", ""),
        "scheduledDate":    req.scheduled_date,
        "startTime":        req.start_time,
        "endTime":          req.end_time,
        "note":             req.note,
        "status":           "scheduled",
        "createdBy":        req.created_by,
        "createdAt":        datetime.now(timezone.utc),
    })

    fcm_token = worker.get("fcmToken", "")
    area = worker.get("assignedArea", "your area")
    _send_fcm(
        token=fcm_token,
        title="Shift Scheduled",
        body=f"Your shift on {req.scheduled_date} starts at {req.start_time}. Report to {area}.",
    )

    return {"success": True, "schedule_id": schedule_id}


@app.get("/admin/shift-schedules")
def admin_shift_schedules(worker_id: Optional[str] = None, date: Optional[str] = None,
                          status: Optional[str] = None, limit: int = 50):
    limit = min(limit, 200)
    q = db.collection("shiftSchedules").order_by("scheduledDate", direction=firestore.Query.DESCENDING)

    if worker_id:
        q = db.collection("shiftSchedules").where(
            filter=FieldFilter("workerId", "==", worker_id)
        ).order_by("scheduledDate", direction=firestore.Query.DESCENDING)

    if status:
        q = db.collection("shiftSchedules").where(
            filter=FieldFilter("status", "==", status)
        ).order_by("scheduledDate", direction=firestore.Query.DESCENDING)

    docs = [d.to_dict() for d in q.limit(limit).stream()]

    if date:
        docs = [d for d in docs if d.get("scheduledDate") == date]

    return {"success": True, "schedules": docs, "count": len(docs)}


@app.delete("/admin/shift-schedules/{schedule_id}")
def admin_delete_schedule(schedule_id: str):
    ref = db.collection("shiftSchedules").document(schedule_id)
    if not ref.get().exists:
        raise HTTPException(status_code=404, detail=f"Schedule '{schedule_id}' not found.")
    ref.delete()
    return {"success": True, "schedule_id": schedule_id}


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
        fill_results = _models["fill"](image)
        results["step3_fill_model"] = f"OK — {len(fill_results[0].boxes)} boxes"
    except Exception as e:
        return {"failed_at": "step3_fill_model", "error": str(e)}
    try:
        waste_results = _models["waste"](image)
        results["step4_waste_model"] = f"OK — {len(waste_results[0].boxes)} boxes"
    except Exception as e:
        return {"failed_at": "step4_waste_model", "error": str(e)}
    try:
        db.collection("bins").document("__test__").set({"ping": True}, merge=True)
        results["step5_firestore"] = "OK"
    except Exception as e:
        return {"failed_at": "step5_firestore", "error": str(e)}
    return {"all_steps": "passed", "details": results}

# ── Analytics Endpoints ───────────────────────────────────────────────────────

@app.get("/admin/analytics/peak-hours")
def analytics_peak_hours(days: int = 30):
    """Pickups grouped by hour of day — shows busiest collection windows."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    hour_counts = [0] * 24
    for doc in db.collection("pickupLogs").where(filter=FieldFilter("completedAt", ">=", cutoff)).stream():
        ts = doc.to_dict().get("completedAt")
        if ts and hasattr(ts, "hour"):
            hour_counts[ts.hour] += 1
    return {
        "success": True,
        "data": [{"hour": f"{h:02d}:00", "collections": hour_counts[h]} for h in range(24)],
    }


@app.get("/admin/analytics/fill-efficiency")
def analytics_fill_efficiency(days: int = 14):
    """Average fill level at pickup per day — lower = collecting too early, higher = collecting too late."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    daily: dict = {}
    for doc in db.collection("pickupLogs").where(filter=FieldFilter("completedAt", ">=", cutoff)).stream():
        data = doc.to_dict()
        ts   = data.get("completedAt")
        fill = data.get("fillLevelAtPickup", 0)
        if ts and hasattr(ts, "month"):
            key = f"{ts.month}/{ts.day}"
            daily.setdefault(key, []).append(fill)
    result = [
        {"date": k, "avgFill": round(sum(v) / len(v), 1)}
        for k, v in sorted(daily.items())
    ]
    return {"success": True, "data": result}


@app.get("/admin/analytics/area-stats")
def analytics_area_stats(days: int = 30):
    """Collections per district — highlights which areas need most service."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    area_counts: dict = {}
    for doc in db.collection("pickupLogs").where(filter=FieldFilter("completedAt", ">=", cutoff)).stream():
        area = doc.to_dict().get("area", "Unknown") or "Unknown"
        area_counts[area] = area_counts.get(area, 0) + 1
    result = sorted(
        [{"area": k, "collections": v} for k, v in area_counts.items()],
        key=lambda x: x["collections"],
        reverse=True,
    )
    return {"success": True, "data": result}


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
