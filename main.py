from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import tempfile
import os
from dotenv import load_dotenv

load_dotenv()

from routes.transcription import router as transcription_router
from routes.subscription import router as subscription_router
from services.firebase_service import verify_token

app = FastAPI(
    title="Elscribe AI API",
    description="Backend API for Elscribe AI transcription app",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(transcription_router)
app.include_router(subscription_router)

@app.get("/")
async def root():
    return {"status": "ok", "service": "Elscribe AI API"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
# force redeploy
