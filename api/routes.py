from fastapi import APIRouter, UploadFile, File, HTTPException
from services.emotion_analysis import analyze_emotion
from services.gemini_prompt import analyze_text_and_decide
from starlette.concurrency import run_in_threadpool
import tempfile
import os
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/analyze")
async def analyze(
    audio_file: UploadFile = File(None),
    text_file: UploadFile = File(...)
):
    try:
        audio_path = None
        if audio_file:
            content = await audio_file.read()
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                tmp.write(content)
                audio_path = tmp.name

        text = (await text_file.read()).decode('utf-8')
        logger.debug(f"Text input: {text[:100]}...")

        result = await run_in_threadpool(analyze_emotion, audio_path, text)
        if audio_path:
            os.remove(audio_path)
            logger.debug("Temporary audio file removed.")

        danger_score = analyze_text_and_decide(text, result['all_emotions'])
        return {"emotion": result, "danger_score": danger_score}
    except Exception as e:
        logger.error("Error in /analyze endpoint", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
