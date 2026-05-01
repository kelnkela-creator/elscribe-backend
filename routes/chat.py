from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
import openai
import os
from services.firebase_service import verify_token

router = APIRouter()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

SUGGESTIONS = [
    "Summarize this transcript",
    "What are the key points?",
    "List action items",
    "Who are the speakers?",
    "What topics were discussed?",
    "What questions were asked?",
]


def get_openai_client():
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")
    return openai.OpenAI(api_key=OPENAI_API_KEY)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    transcript_content: str
    message: str
    history: List[ChatMessage] = []


@router.post("/chat")
async def chat_with_transcript(
    request: ChatRequest,
    user: dict = Depends(verify_token),
):
    client = get_openai_client()

    preview = request.transcript_content[:3500]
    if len(request.transcript_content) > 3500:
        preview += "\n[transcript truncated for brevity]"

    system_prompt = f"""You are a helpful assistant analyzing a transcript for the user.

Transcript:
---
{preview}
---

Answer questions about this transcript concisely and helpfully.
- For summaries: write 2–4 sentences
- For key points or action items: use bullet points (• item)
- For speaker questions: analyze the labels in the transcript
- Keep responses under 250 words unless a detailed analysis is requested
- If the transcript has no relevant content for the question, say so briefly"""

    messages = [{"role": "system", "content": system_prompt}]
    for msg in request.history[-8:]:
        messages.append({"role": msg.role, "content": msg.content})
    messages.append({"role": "user", "content": request.message})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=400,
        temperature=0.7,
    )

    return JSONResponse({
        "reply": response.choices[0].message.content,
        "suggestions": SUGGESTIONS,
    })
