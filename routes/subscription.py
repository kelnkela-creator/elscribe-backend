from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
import stripe
import os
from datetime import datetime, timezone, timedelta
from services.firebase_service import verify_token, db

router = APIRouter()

stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")
STRIPE_PRICE_ID = os.getenv("STRIPE_PRICE_ID")  # $6 / 2 months recurring price


@router.post("/create-subscription")
async def create_subscription(user: dict = Depends(verify_token)):
    uid = user["uid"]
    user_doc = db.collection("users").document(uid).get()
    if not user_doc.exists:
        db.collection("users").document(uid).set({
            "uid": uid,
            "email": user.get("email", ""),
            "subscriptionStatus": "trial",
            "createdAt": datetime.now(timezone.utc),
            "trialStartedAt": datetime.now(timezone.utc),
            "trialEndsAt": datetime.now(timezone.utc) + timedelta(days=14),
            "stripeCustomerId": None,
        })
        user_data = {"stripeCustomerId": None, "email": user.get("email", "")}
    else:
        user_data = user_doc.to_dict()
    customer_id = user_data.get("stripeCustomerId")

    # Create Stripe customer if not exists
    if not customer_id:
        customer = stripe.Customer.create(
            email=user_data.get("email"),
            metadata={"firebase_uid": uid}
        )
        customer_id = customer.id
        db.collection("users").document(uid).update({
            "stripeCustomerId": customer_id
        })

    # Create setup intent for payment method collection
    setup_intent = stripe.SetupIntent.create(
        customer=customer_id,
        payment_method_types=["card"],
        usage="off_session",
    )

    return JSONResponse({
        "clientSecret": setup_intent.client_secret,
        "customerId": customer_id,
    })


@router.post("/confirm-subscription")
async def confirm_subscription(user: dict = Depends(verify_token)):
    uid = user["uid"]
    user_doc = db.collection("users").document(uid).get()
    user_data = user_doc.to_dict()
    customer_id = user_data.get("stripeCustomerId")

    if not customer_id:
        raise HTTPException(status_code=400, detail="No Stripe customer found")

    # Get default payment method
    customer = stripe.Customer.retrieve(customer_id, expand=["default_source"])
    payment_methods = stripe.PaymentMethod.list(
        customer=customer_id, type="card")

    if not payment_methods.data:
        raise HTTPException(status_code=400, detail="No payment method found")

    pm = payment_methods.data[0]
    stripe.Customer.modify(customer_id, invoice_settings={
        "default_payment_method": pm.id
    })

    # Create subscription
    subscription = stripe.Subscription.create(
        customer=customer_id,
        items=[{"price": STRIPE_PRICE_ID}],
        default_payment_method=pm.id,
        trial_end=int((datetime.now(timezone.utc) + timedelta(days=14)).timestamp()),
    )

    # Update Firestore
    db.collection("users").document(uid).update({
        "subscriptionStatus": "active",
        "stripeSubscriptionId": subscription.id,
    })

    return JSONResponse({"status": "subscribed", "subscriptionId": subscription.id})


@router.get("/user-status")
async def get_user_status(user: dict = Depends(verify_token)):
    uid = user["uid"]
    user_doc = db.collection("users").document(uid).get()
    if not user_doc.exists:
        return JSONResponse({"subscriptionStatus": "none"})
    data = user_doc.to_dict() or {}
    status = data.get("subscriptionStatus", "none")
    trial_ends_at = data.get("trialEndsAt")
    if status == "trial" and trial_ends_at:
        if hasattr(trial_ends_at, "timestamp"):
            if trial_ends_at.timestamp() < datetime.now(timezone.utc).timestamp():
                status = "expired"
    return JSONResponse({"subscriptionStatus": status})


@router.post("/start-trial")
async def start_trial(user: dict = Depends(verify_token)):
    uid = user["uid"]
    user_doc = db.collection("users").document(uid).get()
    if user_doc.exists:
        data = user_doc.to_dict() or {}
        current_status = data.get("subscriptionStatus", "none")
        if current_status in ("trial", "active"):
            return JSONResponse({"status": current_status})
        db.collection("users").document(uid).update({
            "subscriptionStatus": "trial",
            "trialStartedAt": datetime.now(timezone.utc),
            "trialEndsAt": datetime.now(timezone.utc) + timedelta(days=14),
        })
    else:
        db.collection("users").document(uid).set({
            "uid": uid,
            "email": user.get("email", ""),
            "subscriptionStatus": "trial",
            "createdAt": datetime.now(timezone.utc),
            "trialStartedAt": datetime.now(timezone.utc),
            "trialEndsAt": datetime.now(timezone.utc) + timedelta(days=14),
            "stripeCustomerId": None,
        })
    return JSONResponse({"status": "trial"})


@router.post("/stripe-webhook")
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, STRIPE_WEBHOOK_SECRET)
    except stripe.error.SignatureVerificationError:
        raise HTTPException(status_code=400, detail="Invalid signature")

    if event["type"] == "customer.subscription.deleted":
        sub = event["data"]["object"]
        customer_id = sub["customer"]
        _update_by_customer(customer_id, {"subscriptionStatus": "cancelled"})

    elif event["type"] == "invoice.payment_failed":
        invoice = event["data"]["object"]
        customer_id = invoice["customer"]
        _update_by_customer(customer_id, {"subscriptionStatus": "past_due"})

    elif event["type"] == "invoice.payment_succeeded":
        invoice = event["data"]["object"]
        customer_id = invoice["customer"]
        _update_by_customer(customer_id, {"subscriptionStatus": "active"})

    return JSONResponse({"received": True})


def _update_by_customer(customer_id: str, updates: dict):
    users = db.collection("users").where(
        "stripeCustomerId", "==", customer_id).stream()
    for u in users:
        db.collection("users").document(u.id).update(updates)
