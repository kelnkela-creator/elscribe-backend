import firebase_admin
from firebase_admin import credentials, auth, firestore
from fastapi import HTTPException, Header
import os
import json

_initialized = False

def _init_firebase():
    global _initialized
    if _initialized:
        return
    service_account_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
    if service_account_json:
        cred_dict = json.loads(service_account_json)
        cred = credentials.Certificate(cred_dict)
    else:
        cred = credentials.Certificate("firebase-service-account.json")
    bucket = os.getenv("FIREBASE_STORAGE_BUCKET", "")
    options = {'storageBucket': bucket} if bucket else {}
    firebase_admin.initialize_app(cred, options)
    _initialized = True

_init_firebase()

db = firestore.client()

async def verify_token(authorization: str = Header(...)) -> dict:
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    token = authorization[7:]
    try:
        decoded = auth.verify_id_token(token)
        return decoded
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

async def get_user_data(uid: str) -> dict:
    try:
        doc = db.collection("users").document(uid).get()
        if not doc.exists:
            return {}
        return doc.to_dict() or {}
    except Exception:
        return {}

async def is_subscription_active(uid: str) -> bool:
    from datetime import datetime, timezone
    data = await get_user_data(uid)
    status = data.get("subscriptionStatus", "none")
    if status == "active":
        return True
    if status == "trial":
        trial_ends = data.get("trialEndsAt")
        if trial_ends:
            if hasattr(trial_ends, 'timestamp'):
                return trial_ends.timestamp() > datetime.now(timezone.utc).timestamp()
        else:
            return True
    return False
