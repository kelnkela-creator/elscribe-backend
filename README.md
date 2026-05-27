# Elscribe AI — Backend API

> FastAPI backend powering the [Elscribe AI](https://www.microsoft.com/store/apps/search?q=elscribe+ai) Windows app — an AI-powered transcription platform available on the Microsoft Store.

---

## Overview

This backend handles all server-side logic for Elscribe AI:

- **Audio/video transcription** using OpenAI Whisper
- **Speaker detection and labeling** using GPT-4o-mini
- **AI chat** for transcript Q&A using GPT-4o-mini
- **Subscription management** via Stripe webhooks
- **User authentication** via Firebase JWT tokens
- **Audio file storage** in Firebase Cloud Storage

The backend is deployed on [Railway](https://railway.app) and serves the Elscribe AI Flutter Windows app.

---

## Architecture

```
Flutter Windows App (Microsoft Store)
        │
        │ HTTPS + Firebase JWT
        ▼
FastAPI Backend (Railway)
  ├── /transcribe   → OpenAI Whisper → timestamped transcript
  ├── /label        → GPT-4o-mini → speaker-labeled transcript
  ├── /chat         → GPT-4o-mini → transcript Q&A
  └── /stripe-webhook → Subscription lifecycle events
        │
        ▼
Firebase (Auth + Firestore + Cloud Storage)
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Framework | FastAPI (Python) |
| Transcription | OpenAI Whisper (`whisper-1`) |
| Speaker Detection | GPT-4o-mini |
| Audio Processing | FFmpeg |
| Auth | Firebase Admin SDK (JWT verification) |
| Database | Firebase Firestore |
| File Storage | Firebase Cloud Storage |
| Payments | Stripe |
| Hosting | Railway |

---

## API Endpoints

### `POST /transcribe`
Transcribes an uploaded audio or video file.

- **Auth:** Firebase JWT (Bearer token)
- **Body:** `multipart/form-data` — `file` (audio/video)
- **Supported formats:** mp3, mp4, wav, m4a, ogg, flac, mov, avi, mkv, webm, mpeg
- **Max file size:** 500MB
- **Returns:** `{ transcript, segments, duration_seconds, filename, audio_url }`

### `POST /label`
Detects and labels speakers in a raw transcript.

- **Auth:** Firebase JWT
- **Body:** `{ "text": "<raw transcript>" }`
- **Returns:** `{ "labeled_transcript": "<formatted with [Speaker 1]:, [Speaker 2]: labels>" }`

### `POST /chat`
Answers questions about a transcript using GPT-4o-mini.

- **Auth:** Firebase JWT
- **Body:** `{ "transcript": "...", "message": "...", "history": [] }`
- **Returns:** `{ "reply": "..." }`

### `POST /stripe-webhook`
Handles Stripe subscription lifecycle events.

- **Events handled:** `customer.subscription.deleted`, `invoice.payment_failed`, `invoice.payment_succeeded`
- **Auth:** Stripe webhook signature (`STRIPE_WEBHOOK_SECRET`)

### `GET /health`
Health check endpoint.

---

## Local Setup

### Prerequisites

- Python 3.10+
- [FFmpeg](https://ffmpeg.org/download.html) installed and in your PATH
- Firebase project with Firestore and Cloud Storage enabled
- OpenAI API key
- Stripe account (for subscription features)

### 1. Clone the repository

```bash
git clone https://github.com/kelnkela-creator/elscribe-backend.git
cd elscribe-backend
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=your_openai_api_key
STRIPE_SECRET_KEY=your_stripe_secret_key
STRIPE_WEBHOOK_SECRET=your_stripe_webhook_signing_secret
FIREBASE_SERVICE_ACCOUNT_JSON={"type":"service_account","project_id":"..."}
FIREBASE_STORAGE_BUCKET=your-project.appspot.com
```

> ⚠️ Never commit your `.env` file. It is included in `.gitignore`.

### 5. Firebase Service Account

Download your Firebase service account JSON from:  
**Firebase Console → Project Settings → Service Accounts → Generate New Private Key**

Paste the entire JSON as the value of `FIREBASE_SERVICE_ACCOUNT_JSON` in your `.env` file (minified to one line), **or** save it as `firebase-service-account.json` in the project root for local development only.

### 6. Run the server

```bash
uvicorn main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`

---

## Deployment (Railway)

This backend is configured to auto-deploy on Railway from the `master` branch.

**Required Railway environment variables:**

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key |
| `STRIPE_SECRET_KEY` | Stripe live/test secret key |
| `STRIPE_WEBHOOK_SECRET` | Stripe webhook signing secret |
| `FIREBASE_SERVICE_ACCOUNT_JSON` | Firebase service account JSON (minified) |
| `FIREBASE_STORAGE_BUCKET` | Firebase Storage bucket name |

---

## Security

- All endpoints (except `/stripe-webhook` and `/health`) require a valid Firebase JWT token
- Subscription status is verified on every transcription request — expired or cancelled users are blocked
- Stripe webhook signatures are verified to prevent fraudulent events
- Audio files are stored with 2-year signed URLs in Firebase Storage
- No API keys or secrets are hardcoded — all loaded via environment variables

---

## Related

- **Elscribe AI Windows App** — Available on the [Microsoft Store](https://www.microsoft.com/store/apps/search?q=elscribe+ai)
- **Ama Tee** — AI agent built on Microsoft Copilot Studio for transcript analysis (Agent Academy Hackathon submission)
- **Landing Page** — [eliscribeai.netlify.app](https://eliscribeai.netlify.app)

---

## License

© 2026 Elijah Eshun. All rights reserved.
