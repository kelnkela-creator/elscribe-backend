import math
import asyncio
import uuid
import json
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import openai
import tempfile
import os
import subprocess
from google.oauth2 import service_account as sa_module
from google.cloud import storage as gcs
from services.firebase_service import verify_token

router = APIRouter()

ALLOWED_EXTENSIONS = {
    'mp3', 'mp4', 'wav', 'm4a', 'ogg', 'flac',
    'mov', 'avi', 'mkv', 'webm', 'mpeg'
}
VIDEO_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv', 'webm', 'mpeg'}
MAX_FILE_SIZE_MB = 500
WHISPER_MAX_MB   = 24    # Whisper API hard limit
CHUNK_DURATION   = 1200  # 20-minute chunks for very long audio
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")


def _upload_audio_to_storage(uid: str, local_path: str) -> str:
    """Upload extracted MP3 to Firebase Storage; return a signed URL valid 2 years."""
    try:
        sa_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON", "{}")
        sa_info = json.loads(sa_json)
        bucket_name = os.getenv("FIREBASE_STORAGE_BUCKET", "")
        if not bucket_name or not sa_info.get('project_id'):
            return 'error:no_config'
        creds = sa_module.Credentials.from_service_account_info(
            sa_info,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        client = gcs.Client(credentials=creds, project=sa_info.get('project_id'))
        bucket = client.bucket(bucket_name)
        blob_name = f'audio/{uid}/{uuid.uuid4()}.mp3'
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(local_path, content_type='audio/mpeg')
        url = blob.generate_signed_url(
            version='v2',
            expiration=timedelta(days=730),
            method='GET',
            credentials=creds,
        )
        return url
    except Exception as e:
        import traceback
        print(f"[AUDIO UPLOAD ERROR] {str(e)}")
        print(traceback.format_exc())
        return f'error:{str(e)[:300]}'


def get_openai_client():
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")
    return openai.OpenAI(api_key=OPENAI_API_KEY)


def _transcribe_single(client, path: str, time_offset: float) -> tuple:
    """Transcribe one file; try verbose_json for timestamps, fall back to text."""
    try:
        with open(path, 'rb') as f:
            response = client.audio.transcriptions.create(
                model="whisper-1", file=f, language="en",
                response_format="verbose_json",
            )
        if isinstance(response, str):
            return response, []
        elif isinstance(response, dict):
            text = response.get('text', '')
            segs = [
                {"start": round(float(s['start']) + time_offset, 2),
                 "end":   round(float(s['end'])   + time_offset, 2),
                 "text":  s['text'].strip()}
                for s in response.get('segments', [])
            ]
            return text, segs
        else:
            text = getattr(response, 'text', '') or ''
            segs = []
            for s in (getattr(response, 'segments', None) or []):
                if isinstance(s, dict):
                    segs.append({"start": round(float(s['start']) + time_offset, 2),
                                 "end":   round(float(s['end'])   + time_offset, 2),
                                 "text":  s['text'].strip()})
                else:
                    segs.append({"start": round(float(s.start) + time_offset, 2),
                                 "end":   round(float(s.end)   + time_offset, 2),
                                 "text":  s.text.strip()})
            return text, segs
    except Exception:
        # Fall back to plain text if verbose_json fails
        with open(path, 'rb') as f:
            response = client.audio.transcriptions.create(
                model="whisper-1", file=f, language="en",
                response_format="text",
            )
        raw = response if isinstance(response, str) else getattr(response, 'text', '')
        return raw, []


class LabelRequest(BaseModel):
    text: str


LABEL_SYSTEM = (
    "You are a transcript formatter. Format with speaker labels.\n"
    "Rules:\n"
    "- Multiple speakers (interview/conversation/Q&A): label as [Speaker 1]:, [Speaker 2]:, etc.\n"
    "- Clear Q&A sections: use [Question]: and [Answer]:\n"
    "- Single speaker (monologue/lecture): return text as-is, no labels\n"
    "- Each speaker segment on its own line\n"
    "- Keep EVERY word exactly unchanged — do NOT skip or summarize anything\n"
    "- Output ONLY the formatted transcript"
)
WORDS_PER_CHUNK = 4000   # large chunks → fewer API calls; 16k output tokens handles this easily


def _label_chunk(client, text: str) -> str:
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": LABEL_SYSTEM},
                      {"role": "user", "content": text}],
            max_tokens=16000,
            temperature=0,
        )
        return r.choices[0].message.content.strip()
    except Exception:
        return text


@router.post("/label")
async def label_transcript(request: LabelRequest, user: dict = Depends(verify_token)):
    client = get_openai_client()
    try:
        words = request.text.split()
        if len(words) <= WORDS_PER_CHUNK:
            # Single call — fast for most transcripts
            labeled = await asyncio.get_event_loop().run_in_executor(
                None, _label_chunk, client, request.text
            )
        else:
            # Split into chunks and process in parallel
            chunks = [
                " ".join(words[i: i + WORDS_PER_CHUNK])
                for i in range(0, len(words), WORDS_PER_CHUNK)
            ]
            with ThreadPoolExecutor() as pool:
                loop = asyncio.get_event_loop()
                results = await asyncio.gather(*[
                    loop.run_in_executor(pool, _label_chunk, client, chunk)
                    for chunk in chunks
                ])
            labeled = "\n".join(results)
        return JSONResponse({"labeled_transcript": labeled})
    except Exception:
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

    extra_paths = []

    try:
        # Get duration
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

        client = get_openai_client()
        raw_text = ''
        segments = []

        # Extract audio as 32kbps mono mp3 (good quality for Whisper, fits ~1.6 hrs in 25MB)
        audio_path = tmp_path + '_audio.mp3'
        extra_paths.append(audio_path)
        audio_ok = False
        ffmpeg_error = ''
        try:
            proc = subprocess.run(
                ['ffmpeg', '-i', tmp_path, '-vn', '-ar', '16000', '-ac', '1',
                 '-b:a', '32k', '-y', audio_path],
                capture_output=True, timeout=300
            )
            ffmpeg_error = proc.stderr.decode('utf-8', errors='replace')[-800:]
            audio_ok = os.path.exists(audio_path) and os.path.getsize(audio_path) > 1000
        except FileNotFoundError:
            ffmpeg_error = 'ffmpeg not found on server'
        except Exception as e:
            ffmpeg_error = str(e)

        if not audio_ok and ext in VIDEO_EXTENSIONS:
            raise HTTPException(
                status_code=422,
                detail=f"Could not extract audio from video file. ffmpeg output: {ffmpeg_error}"
            )

        # Use original file only if ffmpeg failed on a pure-audio format
        transcribe_source = audio_path if audio_ok else tmp_path
        source_mb = os.path.getsize(transcribe_source) / (1024 * 1024)

        uid = user.get('uid', 'unknown')
        loop = asyncio.get_event_loop()

        if source_mb <= WHISPER_MAX_MB:
            # Run Whisper and Firebase Storage upload concurrently for video files
            if audio_ok and ext in VIDEO_EXTENSIONS:
                (raw_text, segments), audio_url = await asyncio.gather(
                    loop.run_in_executor(None, _transcribe_single, client, transcribe_source, 0.0),
                    loop.run_in_executor(None, _upload_audio_to_storage, uid, audio_path),
                )
            else:
                raw_text, segments = await loop.run_in_executor(
                    None, _transcribe_single, client, transcribe_source, 0.0)
                audio_url = ''
        else:
            # File too long: split into 20-minute chunks
            audio_url = ''
            n_chunks = max(1, math.ceil(duration_seconds / CHUNK_DURATION))
            chunk_paths = []
            for i in range(n_chunks):
                start = i * CHUNK_DURATION
                cpath = f'{tmp_path}_chunk_{i}.mp3'
                extra_paths.append(cpath)
                try:
                    subprocess.run(
                        ['ffmpeg', '-i', tmp_path,
                         '-ss', str(start), '-t', str(CHUNK_DURATION),
                         '-vn', '-ar', '16000', '-ac', '1', '-b:a', '32k', '-y', cpath],
                        capture_output=True, timeout=180
                    )
                    if os.path.exists(cpath) and os.path.getsize(cpath) > 500:
                        chunk_paths.append((cpath, float(start)))
                except Exception:
                    pass

            tasks = [loop.run_in_executor(None, _transcribe_single, client, cp, off)
                     for cp, off in chunk_paths]
            if audio_ok and ext in VIDEO_EXTENSIONS:
                tasks.append(loop.run_in_executor(None, _upload_audio_to_storage, uid, audio_path))
                results = await asyncio.gather(*tasks)
                audio_url = results[-1]
                results = results[:-1]
            else:
                results = await asyncio.gather(*tasks)

            all_texts, all_segs = [], []
            for t, s in results:
                all_texts.append(t)
                all_segs.extend(s)
            raw_text = ' '.join(all_texts)
            segments = all_segs

        return JSONResponse({
            "transcript": raw_text,
            "segments": segments,
            "duration_seconds": duration_seconds,
            "filename": file.filename,
            "audio_url": audio_url,
        })

    except openai.APIError as e:
        raise HTTPException(status_code=502, detail=f"Transcription service error: {str(e)}")
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        for p in extra_paths:
            try:
                if p and os.path.exists(p):
                    os.unlink(p)
            except Exception:
                pass
