"""
seed_firestore.py — Populate Firestore with realistic Berlin test data.

Usage:
    python seed_firestore.py                  # seed everything
    python seed_firestore.py --only bins      # only bins
    python seed_firestore.py --only workers   # only workers
    python seed_firestore.py --clear          # delete existing docs first
"""

import argparse
import sys
from datetime import datetime, timezone

import firebase_admin
from firebase_admin import auth, credentials, firestore

# ── Firebase init ─────────────────────────────────────────────────────────────
SERVICE_ACCOUNT = "serviceAccountKey.json"
try:
    cred = credentials.Certificate(SERVICE_ACCOUNT)
    firebase_admin.initialize_app(cred)
except Exception as e:
    print(f"[ERROR] Could not initialize Firebase: {e}")
    print(f"        Make sure '{SERVICE_ACCOUNT}' is in the Backend/ directory.")
    sys.exit(1)

db = firestore.client()
now = datetime.now(timezone.utc)

DEMO_PASSWORD = "CleanCore@123"

# ── Bin test data — 21 bins across Islamabad/Rawalpindi ──────────────────────
# IMPORTANT: area strings here must be BYTE-FOR-BYTE identical to the
# assignedArea values in seed_workers.py (exact match, case-sensitive).
#
# fillLevel, fillStatus, and aiConfidence are real values produced by running
# the YOLO fill_model + waste_model on the 21 images in asset/bins/ (1-to-1
# mapping in alphabetical filename order).  Run infer_bins.py to regenerate.
BINS: list[dict] = [
    # ── G-9 Markaz, Islamabad ─────────────────────────────────────────────────
    # image: 1000638068_jpg.rf.5990179f6797b8de122dd9b42b00d11a.jpg
    {"area": "G-9",  "wasteType": "General", "fillLevel": 98, "fillStatus": "Overflowing", "aiConfidence": 0.971, "lat": 33.6938, "lng": 73.0651},
    # image: IMG-20251004-WA0012_jpg.rf.df3adb70ceccc6b8c365210c6a26dd1f.jpg
    {"area": "G-9",  "wasteType": "Plastic",  "fillLevel": 55, "fillStatus": "Half-Full",   "aiConfidence": 0.855, "lat": 33.6850, "lng": 73.0580},
    # image: IMG-20251004-WA0029_jpg.rf.e15d76723cb76cd76900b82fc934f953.jpg
    {"area": "G-9",  "wasteType": "Organic",  "fillLevel": 98, "fillStatus": "Overflowing", "aiConfidence": 0.684, "lat": 33.6880, "lng": 73.0620},
    # image: IMG-20251004-WA0032-1-_jpg.rf.4886555c49a49e31e0d458d4b09fad0b.jpg
    {"area": "G-9",  "wasteType": "General",  "fillLevel": 98, "fillStatus": "Overflowing", "aiConfidence": 0.931, "lat": 33.6910, "lng": 73.0700},

    # ── Blue Area, Islamabad ──────────────────────────────────────────────────
    # image: IMG-20251016-WA0003_jpg.rf.eeacaeb7b42ac33ce8d570ccda5c5f06.jpg
    {"area": "Blue Area", "wasteType": "General", "fillLevel": 55, "fillStatus": "Half-Full", "aiConfidence": 0.948, "lat": 33.7264, "lng": 73.0979},
    # image: IMG-20251016-WA0004_jpg.rf.abc0e7dac0881d708de5235e89c81300.jpg
    {"area": "Blue Area", "wasteType": "Plastic",  "fillLevel": 55, "fillStatus": "Half-Full", "aiConfidence": 0.945, "lat": 33.7240, "lng": 73.0940},
    # image: IMG-20251016-WA0008_jpg.rf.62dec5bb9e357db1396783409c3b91e7.jpg
    {"area": "Blue Area", "wasteType": "Organic",  "fillLevel": 55, "fillStatus": "Half-Full", "aiConfidence": 0.925, "lat": 33.7280, "lng": 73.1010},
    # image: IMG-20251016-WA0018_jpg.rf.003c60b2e5d5a605bee9f805e51888b1.jpg
    {"area": "Blue Area", "wasteType": "Paper",    "fillLevel": 85, "fillStatus": "Full",       "aiConfidence": 0.962, "lat": 33.7260, "lng": 73.0960},

    # ── F-6 Islamabad ─────────────────────────────────────────────────────────
    # image: IMG-20251016-WA0020_jpg.rf.0d725669fe9e26dc382465d2ad125736.jpg
    {"area": "F-6 Islamabad", "wasteType": "Plastic", "fillLevel": 85, "fillStatus": "Full",      "aiConfidence": 0.960, "lat": 33.7296, "lng": 73.0748},
    # image: IMG_20251014_163927_jpg.rf.a32c8967584f3ca68d66f5c43c96ada1.jpg
    {"area": "F-6 Islamabad", "wasteType": "Plastic", "fillLevel": 55, "fillStatus": "Half-Full", "aiConfidence": 0.926, "lat": 33.7315, "lng": 73.0780},

    # ── Saddar Pindi ──────────────────────────────────────────────────────────
    # image: IMG_20251014_164011_jpg.rf.6d48f4b2185c6003be8cd2d23bdf434a.jpg
    {"area": "Saddar Pindi", "wasteType": "Glass", "fillLevel": 55, "fillStatus": "Half-Full", "aiConfidence": 0.953, "lat": 33.5950, "lng": 73.0545},
    # image: IMG_20251014_164205_jpg.rf.8258881cd4066c9fe4182083fc41d9e3.jpg
    {"area": "Saddar Pindi", "wasteType": "Glass", "fillLevel": 55, "fillStatus": "Half-Full", "aiConfidence": 0.967, "lat": 33.5910, "lng": 73.0500},
    # image: IMG_20251014_170013_1_jpg.rf.cb19c691c8faa0ce74d134908ac0ed64.jpg
    {"area": "Saddar Pindi", "wasteType": "Glass", "fillLevel":  5, "fillStatus": "Empty",     "aiConfidence": 0.899, "lat": 33.5980, "lng": 73.0590},

    # ── E-11 Islamabad ────────────────────────────────────────────────────────
    # image: IMG_20251025_140150_jpg.rf.8905517a01bf92a34bd15c51a77e7b47.jpg
    {"area": "E-11 Islamabad", "wasteType": "Metal", "fillLevel": 85, "fillStatus": "Full",      "aiConfidence": 0.850, "lat": 33.6995, "lng": 72.9830},
    # image: IMG_20251025_140757_jpg.rf.8fd5432385f23c8cf99fd5af46f54d9f.jpg
    {"area": "E-11 Islamabad", "wasteType": "Metal", "fillLevel": 85, "fillStatus": "Full",      "aiConfidence": 0.918, "lat": 33.7020, "lng": 72.9870},
    # image: IMG_20251025_143802_jpg.rf.bd858ece8f2103e8b3b9d8c872f26c51.jpg
    {"area": "E-11 Islamabad", "wasteType": "Metal", "fillLevel": 85, "fillStatus": "Full",      "aiConfidence": 0.941, "lat": 33.6970, "lng": 72.9790},

    # ── Bahria Phase 7 ────────────────────────────────────────────────────────
    # image: IMG_20251025_143947_jpg.rf.de3df66ddc2d817497f67c37e17672ff.jpg
    {"area": "Bahria Phase 7", "wasteType": "Organic", "fillLevel": 55, "fillStatus": "Half-Full", "aiConfidence": 0.960, "lat": 33.5235, "lng": 73.0945},
    # image: IMG_20251025_144541_jpg.rf.12fc8f3fcf890c32cea5aa939fafbb5b.jpg
    {"area": "Bahria Phase 7", "wasteType": "Organic", "fillLevel":  5, "fillStatus": "Empty",     "aiConfidence": 0.901, "lat": 33.5250, "lng": 73.0980},

    # ── I-8 Islamabad ─────────────────────────────────────────────────────────
    # image: IMG_20251025_145109_jpg.rf.1f47bfafcace29c9b4a3cda35170fbd6.jpg
    {"area": "I-8 Islamabad", "wasteType": "Paper", "fillLevel": 55, "fillStatus": "Half-Full", "aiConfidence": 0.928, "lat": 33.6685, "lng": 73.0765},
    # image: IMG_20251025_145725_jpg.rf.ba7a1a84250d5f36a75e88025f5fed5f.jpg
    {"area": "I-8 Islamabad", "wasteType": "Paper", "fillLevel": 85, "fillStatus": "Full",      "aiConfidence": 0.866, "lat": 33.6610, "lng": 73.0640},
    # image: IMG_20251025_152731_jpg.rf.3d5ce5d5132395d5e0be4f0edc972fee.jpg
    {"area": "I-8 Islamabad", "wasteType": "Paper", "fillLevel": 55, "fillStatus": "Half-Full", "aiConfidence": 0.902, "lat": 33.6710, "lng": 73.0800},
]

# ── Worker test accounts — must match an existing Firebase Auth UID or be      ──
# ── created manually. uid values here are placeholders; swap them for real    ──
# ── UIDs from your Firebase Auth console.                                     ──
WORKERS: list[dict] = [
    # F-6 Islamabad — Plastic collection
    {
        "uid": "worker_f6_001",
        "email": "cleancore.worker.f6@gmail.com",
        "firstName": "Ali",
        "lastName": "Hassan",
        "role": "worker",
        "status": "active",
        "assignedArea": "F-6 Islamabad",
        "assignedWasteType": "Plastic",
        "lat": 33.7296,
        "lng": 73.0748,
        "profilePicture": "",
    },
    # G-9 Islamabad — General waste
    {
        "uid": "worker_g9_001",
        "email": "cleancore.worker.g9@gmail.com",
        "firstName": "Sara",
        "lastName": "Malik",
        "role": "worker",
        "status": "active",
        "assignedArea": "G-9 Islamabad",
        "assignedWasteType": "General",
        "lat": 33.6844,
        "lng": 73.0238,
        "profilePicture": "",
    },
    # Bahria Phase 7 — Organic waste
    {
        "uid": "worker_bahria_001",
        "email": "cleancore.worker.bahria@gmail.com",
        "firstName": "Omar",
        "lastName": "Farooq",
        "role": "worker",
        "status": "active",
        "assignedArea": "Bahria Phase 7",
        "assignedWasteType": "Organic",
        "lat": 33.5235,
        "lng": 73.0945,
        "profilePicture": "",
    },
    # Saddar Pindi — Glass collection
    {
        "uid": "worker_saddar_001",
        "email": "cleancore.worker.saddar@gmail.com",
        "firstName": "Zara",
        "lastName": "Ahmed",
        "role": "worker",
        "status": "active",
        "assignedArea": "Saddar Pindi",
        "assignedWasteType": "Glass",
        "lat": 33.5950,
        "lng": 73.0545,
        "profilePicture": "",
    },
    # E-11 Islamabad — Metal collection
    {
        "uid": "0b4s20czGicUJTeWUfvO1df7yCM2",
        "email": "bilal.khan@gmail.com",
        "firstName": "Bilal",
        "lastName": "Khan",
        "role": "worker",
        "status": "active",
        "assignedArea": "E-11 Islamabad",
        "assignedWasteType": "Metal",
        "lat": 33.6995,
        "lng": 72.9830,
        "profilePicture": "",
    },
    # I-8 Islamabad — Paper collection
    {
        "uid": "worker_i8_001",
        "email": "cleancore.worker.i8@gmail.com",
        "firstName": "Hina",
        "lastName": "Raza",
        "role": "worker",
        "status": "active",
        "assignedArea": "I-8 Islamabad",
        "assignedWasteType": "Paper",
        "lat": 33.6685,
        "lng": 73.0765,
        "profilePicture": "",
    },
]

ADMIN_USER = {
    "uid": "admin_cleancore_001",
    "email": "cleancore.admin@gmail.com",
    "firstName": "Admin",
    "lastName": "CleanCore",
    "role": "admin",
    "status": "active",
    "assignedArea": "",
    "assignedWasteType": "",
    "lat": 33.7215,
    "lng": 73.0433,
    "profilePicture": "",
}


def _status_from_fill(fill: int) -> str:
    if fill >= 90:
        return "full"
    if fill >= 70:
        return "partial"
    return "empty"


def clear_collection(col: str) -> None:
    docs = db.collection(col).stream()
    batch = db.batch()
    count = 0
    for doc in docs:
        batch.delete(doc.reference)
        count += 1
        if count % 499 == 0:
            batch.commit()
            batch = db.batch()
    if count % 499 != 0:
        batch.commit()
    print(f"  Deleted {count} docs from '{col}'")


def seed_bins() -> None:
    print("\n[bins] Seeding bins...")
    batch = db.batch()
    for b in BINS:
        ref = db.collection("bins").document()
        batch.set(ref, {
            "area":          b["area"],
            "wasteType":     b["wasteType"],
            "fillLevel":     b["fillLevel"],
            "fillStatus":    b["fillStatus"],
            "status":        _status_from_fill(b["fillLevel"]),
            "lat":           b["lat"],
            "lng":           b["lng"],
            "isLocked":      False,
            "capacity":      240,
            "aiConfidence":  b["aiConfidence"],
            "lastAnalyzed":  now,
            "createdAt":     now,
        })
    batch.commit()
    print(f"  Seeded {len(BINS)} bins.")


def _upsert_auth(uid: str, email: str, display_name: str, password: str) -> str:
    """Create or update a Firebase Auth account. Returns the final UID."""
    try:
        auth.create_user(
            uid=uid,
            email=email,
            password=password,
            display_name=display_name,
            email_verified=True,
        )
        print(f"    ✓ Created  auth: {email}")
    except auth.UidAlreadyExistsError:
        auth.update_user(uid, email=email, display_name=display_name)
        print(f"    ~ Updated  auth: {email} (UID already existed)")
    except auth.EmailAlreadyExistsError:
        # Email belongs to a different UID — look it up so Firestore is consistent
        existing = auth.get_user_by_email(email)
        print(f"    ! Conflict auth: {email} → using existing UID {existing.uid}")
        return existing.uid
    return uid


def seed_workers() -> None:
    print("\n[auth] Creating Firebase Auth accounts...")
    all_accounts = [
        *WORKERS,
        ADMIN_USER,
    ]
    uid_map: dict[str, str] = {}  # original uid → resolved uid
    for acct in all_accounts:
        display = f"{acct.get('firstName', '')} {acct.get('lastName', '')}".strip()
        resolved_uid = _upsert_auth(
            uid=acct["uid"],
            email=acct["email"],
            display_name=display,
            password=DEMO_PASSWORD,
        )
        uid_map[acct["uid"]] = resolved_uid

    print(f"\n[users] Seeding Firestore user documents...")
    batch = db.batch()
    for w in WORKERS:
        real_uid = uid_map.get(w["uid"], w["uid"])
        ref = db.collection("users").document(real_uid)
        batch.set(ref, {**w, "uid": real_uid, "createdAt": now}, merge=True)
    real_admin_uid = uid_map.get(ADMIN_USER["uid"], ADMIN_USER["uid"])
    ref = db.collection("users").document(real_admin_uid)
    batch.set(ref, {**ADMIN_USER, "uid": real_admin_uid, "createdAt": now}, merge=True)
    batch.commit()

    print(f"\n  Seeded {len(WORKERS)} workers + 1 admin.")
    print()
    print("  ── Login credentials (password for all accounts) ──────────────")
    print(f"  Password : {DEMO_PASSWORD}")
    print()
    for acct in all_accounts:
        role = acct.get("role", "worker").upper()
        print(f"  [{role:6}] {acct['email']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed Firestore with Berlin test data")
    parser.add_argument("--only", choices=["bins", "workers"], help="Seed only one collection")
    parser.add_argument("--clear", action="store_true", help="Delete existing documents before seeding")
    args = parser.parse_args()

    do_bins    = args.only in (None, "bins")
    do_workers = args.only in (None, "workers")

    if args.clear:
        print("\n[clear] Deleting existing data...")
        if do_bins:
            clear_collection("bins")
        if do_workers:
            clear_collection("users")

    if do_bins:
        seed_bins()
    if do_workers:
        seed_workers()

    print("\nDone. Refresh the Admin Panel to see the seeded data.")


if __name__ == "__main__":
    main()
