from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import openai
import tempfile
import os
import subprocess
from services.firebase_service import verify_token

router = APIRouter()

ALLOWED_EXTENSIONS = {
    'mp3', 'mp4', 'wav', 'm4a', 'ogg', 'flac',
    'mov', 'avi', 'mkv', 'webm', 'mpeg'
}
MAX_FILE_SIZE_MB = 500
WHISPER_MAX_MB = 24  # Whisper API limit is 25MB
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def get_openai_client():
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")
    return openai.OpenAI(api_key=OPENAI_API_KEY)


class LabelRequest(BaseModel):
    text: str


@router.post("/label")
async def label_transcript(request: LabelRequest, user: dict = Depends(verify_token)):
    """Add speaker/Q&A labels to a raw transcript using GPT."""
    client = get_openai_client()
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a transcript formatter. Format with speaker labels.\n"
                        "Rules:\n"
                        "- Multiple speakers (interview/conversation/Q&A): label as [Speaker 1]:, [Speaker 2]:, etc.\n"
                        "- Clear Q&A sections: use [Question]: and [Answer]:\n"
                        "- Single speaker (monologue/lecture): return text as-is, no labels\n"
                        "- Each speaker segment on its own line\n"
                        "- Keep original words exactly unchanged\n"
                        "- Output ONLY the formatted transcript"
                    ),
                },
                {"role": "user", "content": request.text},
            ],
            max_tokens=3000,
            temperature=0,
        )
        labeled = response.choices[0].message.content.strip()
        return JSONResponse({"labeled_transcript": labeled})
    except Exception as e:
        return JSONResponse({"labeled_transcript": request.text})


@router.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    user: dict = Depends(verify_token),
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

    compressed_path = None

    try:
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

        # Compress to mp3 if file exceeds Whisper's 25MB limit
        transcribe_path = tmp_path
        if size_mb > WHISPER_MAX_MB:
            try:
                compressed_path = tmp_path + '_compressed.mp3'
                subprocess.run(
                    ['ffmpeg', '-i', tmp_path, '-vn', '-ar', '16000', '-ac', '1',
                     '-b:a', '32k', '-y', compressed_path],
                    capture_output=True, timeout=300
                )
                if os.path.exists(compressed_path):
                    transcribe_path = compressed_path
            except Exception:
                pass  # fall back to original file

        client = get_openai_client()
        with open(transcribe_path, 'rb') as audio_file:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="en",
                response_format="verbose_json",
            )

        raw_text = response.text

        # Extract time-aligned segments for audio highlighting in the app
        segments = []
        if hasattr(response, 'segments') and response.segments:
            segments = [
                {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
                for s in response.segments
            ]

        return JSONResponse({
            "transcript": raw_text,
            "segments": segments,
            "duration_seconds": duration_seconds,
            "filename": file.filename,
        })

    except openai.APIError as e:
        raise HTTPException(status_code=502, detail=f"Transcription service error: {str(e)}")
    finally:
        os.unlink(tmp_path)
        if compressed_path and os.path.exists(compressed_path):
            try:
                os.unlink(compressed_path)
            except Exception:
                pass
