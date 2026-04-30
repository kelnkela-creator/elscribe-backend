from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
import openai
import tempfile
import os
import subprocess
from services.firebase_service import verify_token, is_subscription_active

router = APIRouter()

ALLOWED_EXTENSIONS = {
    'mp3', 'mp4', 'wav', 'm4a', 'ogg', 'flac',
    'mov', 'avi', 'mkv', 'webm', 'mpeg'
}
MAX_FILE_SIZE_MB = 500
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def get_openai_client():
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")
    return openai.OpenAI(api_key=OPENAI_API_KEY)


async def check_subscription(user: dict = Depends(verify_token)):
    uid = user["uid"]
    active = await is_subscription_active(uid)
    if not active:
        raise HTTPException(
            status_code=403,
            detail="No active subscription. Please subscribe to continue."
        )
    return user


@router.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    user: dict = Depends(check_subscription),
):
    ext = file.filename.split('.')[-1].lower() if file.filename else ''
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")

    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(status_code=413,
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE_MB}MB")

    with tempfile.NamedTemporaryFile(suffix=f'.{ext}', delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Get duration using ffprobe if available
        duration_seconds = 0
        try:
            result = subprocess.run(
                ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                 '-of', 'default=noprint_wrappers=1:nokey=1', tmp_path],
                capture_output=True, text=True, timeout=30
            )
            duration_seconds = int(float(result.stdout.strip()))
        except Exception:
            pass

        # Transcribe with Whisper
        client = get_openai_client()
        with open(tmp_path, 'rb') as audio_file:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="en",
                response_format="text"
            )

        transcript_text = response if isinstance(response, str) else response.text

        return JSONResponse({
            "transcript": transcript_text,
            "duration_seconds": duration_seconds,
            "filename": file.filename,
        })

    except openai.APIError as e:
        raise HTTPException(status_code=502, detail=f"Transcription service error: {str(e)}")
    finally:
        os.unlink(tmp_path)
