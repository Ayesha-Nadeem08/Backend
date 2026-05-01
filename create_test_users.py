"""
create_test_users.py — Create Firebase Auth accounts + Firestore profiles
for testing the CleanCore Flutter app.

Usage:
    python create_test_users.py

This script:
  1. Creates Firebase Auth accounts (email/password) for each test user
  2. Writes matching Firestore 'users' documents so the app can load profiles
  3. Prints credentials you can use to log in

NOTE: Your Flutter app requires @gmail.com emails (auth_service.dart line 35)
      and email verification (line 48). This script uses Firebase Admin SDK
      which bypasses both checks — but the Flutter login screen still enforces
      the @gmail.com rule. So we use @gmail.com emails for the test accounts.
"""

import sys
from datetime import datetime, timezone

import firebase_admin
from firebase_admin import credentials, firestore, auth

# ── Firebase init ─────────────────────────────────────────────────────────────
SERVICE_ACCOUNT = "serviceAccountKey.json"
try:
    cred = credentials.Certificate(SERVICE_ACCOUNT)
    firebase_admin.initialize_app(cred)
except ValueError:
    # Already initialized (e.g. running in same process as seed_firestore)
    pass
except Exception as e:
    print(f"[ERROR] Could not initialize Firebase: {e}")
    sys.exit(1)

db = firestore.client()
now = datetime.now(timezone.utc)

# ── Test users ────────────────────────────────────────────────────────────────
# Password for ALL test accounts (change as needed)
DEFAULT_PASSWORD = "Test123456"

TEST_USERS = [
    {
        "email": "hans.worker@gmail.com",
        "password": DEFAULT_PASSWORD,
        "firstName": "Hans",
        "lastName": "Müller",
        "role": "worker",
        "status": "active",
        "assignedArea": "F-6 Islamabad",
        "assignedWasteType": "Plastic",
        "lat": 33.7296,
        "lng": 73.0748,
    },
    {
        "email": "anna.worker@gmail.com",
        "password": DEFAULT_PASSWORD,
        "firstName": "Anna",
        "lastName": "Schmidt",
        "role": "worker",
        "status": "active",
        "assignedArea": "G-9 Islamabad",
        "assignedWasteType": "General",
        "lat": 33.6844,
        "lng": 73.0238,
    },
    {
        "email": "felix.worker@gmail.com",
        "password": DEFAULT_PASSWORD,
        "firstName": "Felix",
        "lastName": "Bauer",
        "role": "worker",
        "status": "active",
        "assignedArea": "Rawalpindi Cantt",
        "assignedWasteType": "Organic",
        "lat": 33.5950,
        "lng": 73.0545,
    },
    {
        "email": "admin.cleancore@gmail.com",
        "password": DEFAULT_PASSWORD,
        "firstName": "Admin",
        "lastName": "CleanCore",
        "role": "admin",
        "status": "active",
        "assignedArea": "Islamabad",
        "assignedWasteType": "All",
        "lat": 33.7077,
        "lng": 73.0499,
    },
]


def create_user(user_data: dict) -> str | None:
    """Create a Firebase Auth account and return the UID."""
    email = user_data["email"]
    password = user_data["password"]

    # Check if account already exists
    try:
        existing = auth.get_user_by_email(email)
        print(f"  [OK] Auth account already exists: {email} (uid: {existing.uid})")
        return existing.uid
    except auth.UserNotFoundError:
        pass

    # Create new Auth account
    try:
        user_record = auth.create_user(
            email=email,
            password=password,
            display_name=f"{user_data['firstName']} {user_data['lastName']}",
            email_verified=True,  # Mark as verified so the app lets them in
        )
        print(f"  [OK] Created Auth account: {email} (uid: {user_record.uid})")
        return user_record.uid
    except Exception as e:
        print(f"  [FAIL] Failed to create {email}: {e}")
        return None


def create_firestore_profile(uid: str, user_data: dict) -> None:
    """Write the user profile document to Firestore 'users' collection."""
    doc_data = {
        "uid": uid,
        "email": user_data["email"],
        "firstName": user_data["firstName"],
        "lastName": user_data["lastName"],
        "role": user_data.get("role", "worker"),
        "status": user_data.get("status", "active"),
        "assignedArea": user_data.get("assignedArea", ""),
        "assignedWasteType": user_data.get("assignedWasteType", ""),
        "lat": user_data.get("lat", 0.0),
        "lng": user_data.get("lng", 0.0),
        "profilePicture": "",
        "createdAt": now,
    }
    db.collection("users").document(uid).set(doc_data, merge=True)
    print(f"  [OK] Firestore profile written for {uid}")


def main():
    print("=" * 60)
    print("  CleanCore — Test User Setup")
    print("=" * 60)

    results = []

    for user_data in TEST_USERS:
        print(f"\n[{user_data['role'].upper()}] {user_data['firstName']} {user_data.get('lastName', '')}")
        uid = create_user(user_data)
        if uid:
            create_firestore_profile(uid, user_data)
            results.append((user_data["email"], user_data["password"], user_data["role"], uid))

    # Print summary
    print("\n" + "=" * 60)
    print("  LOGIN CREDENTIALS")
    print("=" * 60)
    print(f"\n  Password for ALL accounts: {DEFAULT_PASSWORD}\n")
    print(f"  {'Email':<30} {'Role':<8} {'UID'}")
    print(f"  {'-'*28}   {'-'*6}   {'-'*28}")
    for email, pwd, role, uid in results:
        print(f"  {email:<30} {role:<8} {uid}")

    print(f"\n  To log in on the Flutter app:")
    print(f"  1. Open the app on Chrome or your device")
    print(f"  2. Enter email: hans.worker@gmail.com")
    print(f"  3. Enter password: {DEFAULT_PASSWORD}")
    print(f"  4. Tap 'Log in'")
    print(f"\n  The app will load the Home page -> tap 'Map' to see bins.\n")


if __name__ == "__main__":
    main()
