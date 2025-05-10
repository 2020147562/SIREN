from fastapi import APIRouter, UploadFile, File
from typing import Optional
from services.emotion_analysis import analyze_audio_emotion
from services.gemini_prompt import analyze_text_and_decide
import tempfile
import os
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/analyze")
async def analyze(audio_file: Optional[UploadFile] = File(None), text_file: UploadFile = File(...)):
    try:
        # Save audio to temp file
        if audio_file:
            logger.debug(f"Processing audio file: {audio_file.filename}")
            audio_bytes = await audio_file.read()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                f.write(audio_bytes)
                audio_path = f.name
                logger.debug(f"Audio saved to temp file: {audio_path}")
        else:
            audio_path = None
            logger.debug("No audio file provided")

        # Load text
        text = (await text_file.read()).decode("utf-8")
        logger.debug(f"Text loaded: {text[:100]}...")  # 처음 100자만 로깅
        
        # 감정 분석
        emotion_result = {}
        if audio_path:
            logger.debug("Starting emotion analysis")
            analysis_result = analyze_audio_emotion(audio_path)
            emotion_result = analysis_result["all_emotions"]  # all_emotions만 전달
            logger.debug(f"Emotion analysis result: {emotion_result}")
            os.remove(audio_path)
            logger.debug("Temporary audio file removed")

        # Gemini 분석 (RAG 포함)
        logger.debug("Starting Gemini analysis")
        danger_score, prompt = analyze_text_and_decide(text, emotion_result)
        logger.debug(f"Danger score: {danger_score}")

        return {
            "emotion": emotion_result,
            "danger_score": danger_score
        }
    except Exception as e:
        logger.error(f"Error in analyze endpoint: {str(e)}", exc_info=True)
        raise
