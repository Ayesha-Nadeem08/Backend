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

# ── Bin test data — 18 bins across 6 Berlin districts ─────────────────────────
BINS: list[dict] = [
    # Islamabad - F-6/F-7 area
    {"area": "F-6 Islamabad", "wasteType": "Plastic", "fillLevel": 92, "lat": 33.7296, "lng": 73.0748},
    {"area": "F-7 Islamabad", "wasteType": "Plastic", "fillLevel": 74, "lat": 33.7226, "lng": 73.0583},
    {"area": "F-6 Islamabad", "wasteType": "Plastic", "fillLevel": 35, "lat": 33.7315, "lng": 73.0780},

    # Islamabad - G-9 area
    {"area": "G-9 Islamabad", "wasteType": "General", "fillLevel": 88, "lat": 33.6844, "lng": 73.0238},
    {"area": "G-10 Islamabad", "wasteType": "General", "fillLevel": 91, "lat": 33.6766, "lng": 73.0112},
    {"area": "G-9 Islamabad", "wasteType": "General", "fillLevel": 55, "lat": 33.6880, "lng": 73.0270},

    # Rawalpindi - Saddar/Cantt area
    {"area": "Saddar Pindi", "wasteType": "Glass", "fillLevel": 78, "lat": 33.5950, "lng": 73.0545},
    {"area": "Pindi Cantt", "wasteType": "Glass", "fillLevel": 95, "lat": 33.5910, "lng": 73.0500},
    {"area": "Saddar Pindi", "wasteType": "Glass", "fillLevel": 42, "lat": 33.5980, "lng": 73.0590},

    # Islamabad - E-11 area
    {"area": "E-11 Islamabad", "wasteType": "Metal", "fillLevel": 85, "lat": 33.6995, "lng": 72.9830},
    {"area": "E-11 Islamabad", "wasteType": "Metal", "fillLevel": 70, "lat": 33.7020, "lng": 72.9870},
    {"area": "E-11 Islamabad", "wasteType": "Metal", "fillLevel": 28, "lat": 33.6970, "lng": 72.9790},

    # Rawalpindi - Bahria Phase 7
    {"area": "Bahria Phase 7", "wasteType": "Organic", "fillLevel": 93, "lat": 33.5235, "lng": 73.0945},
    {"area": "Bahria Phase 7", "wasteType": "Organic", "fillLevel": 76, "lat": 33.5250, "lng": 73.0980},
    {"area": "Bahria Phase 8", "wasteType": "Organic", "fillLevel": 60, "lat": 33.5180, "lng": 73.0850},

    # Islamabad - I-8/I-9 area
    {"area": "I-8 Islamabad", "wasteType": "Paper", "fillLevel": 89, "lat": 33.6685, "lng": 73.0765},
    {"area": "I-9 Islamabad", "wasteType": "Paper", "fillLevel": 71, "lat": 33.6610, "lng": 73.0640},
    {"area": "I-8 Islamabad", "wasteType": "Paper", "fillLevel": 45, "lat": 33.6710, "lng": 73.0800},
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
    for i, b in enumerate(BINS):
        ref = db.collection("bins").document()
        batch.set(ref, {
            "area":          b["area"],
            "wasteType":     b["wasteType"],
            "fillLevel":     b["fillLevel"],
            "status":        _status_from_fill(b["fillLevel"]),
            "lat":           b["lat"],
            "lng":           b["lng"],
            "isLocked":      False,
            "capacity":      240,
            "aiConfidence":  0.0,
            "lastAnalyzed":  None,
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
