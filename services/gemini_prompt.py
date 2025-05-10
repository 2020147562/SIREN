from fastapi import APIRouter, UploadFile, File, HTTPException
from services.emotion_analysis import analyze_emotion
from starlette.concurrency import run_in_threadpool
import tempfile
import os
import logging
import google.generativeai as genai

logger = logging.getLogger(__name__)
router = APIRouter()

# Configure the Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

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

        danger_score, prompt = analyze_text_and_decide(text, result['all_emotions'])
        return {"emotion": result, "danger_score": danger_score}
    except Exception as e:
        logger.error("Error in /analyze endpoint", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

def analyze_text_and_decide(text: str, emotion_results: dict) -> float:
    """
    Analyze text and emotion results to determine danger score.
    Returns a float between 0-100.
    """
    try:
        # Create prompt
        prompt = f"""
        Analyze the following text and emotion analysis results to determine a danger score (0-100).
        Return ONLY a single number between 0-100.

        Text: {text}
        Emotion Analysis: {emotion_results}

        Danger Score Criteria:
        0-59: Low risk - Normal conversation or positive emotions
        60-74: Moderate risk - Some concerning elements
        75-89: High risk - Significant concerning elements
        90-100: Critical risk - Immediate attention required

        Return ONLY the number:
        """

        # Generate response
        model = genai.GenerativeModel('gemini-1.0-pro')
        response = model.generate_content(prompt)
        
        # Extract number from response
        try:
            # Try to find a number in the response
            import re
            numbers = re.findall(r'\d+', response.text)
            if numbers:
                score = float(numbers[0])
                return min(max(score, 0), 100)  # Ensure score is between 0-100
        except:
            pass
        
        return 0.0  # Default score if no number found
        
    except Exception as e:
        logger.error(f"Error in text analysis: {str(e)}")
        return 0.0