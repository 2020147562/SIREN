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
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Check available models during startup
try:
    models = genai.list_models()
    logger.info("=== Available Gemini Models ===")
    for model in models:
        logger.info(f"Model: {model.name}")
        logger.info(f"Display name: {model.display_name}")
        logger.info(f"Description: {model.description}")
        logger.info(f"Generation methods: {model.supported_generation_methods}")
        logger.info("---")
except Exception as e:
    logger.error(f"Error checking available models: {str(e)}")

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
        You are an expert crisis intervention specialist with 20 years of experience in violence prevention and risk assessment.
        Your primary role is to analyze conversations for potential physical or mental violence risks.
        You have successfully prevented numerous violent incidents through your careful analysis and quick response.

        TASK:
        Analyze the following conversation and emotional data to assess the risk level of physical or mental violence.
        Your analysis will directly impact whether the system should notify trusted contacts or authorities.

        INPUT DATA:
        [CONVERSATION RECORD]
        {text}

        [EMOTION ANALYSIS]
        This data represents the emotional state of participants, derived from:
        - Voice tone analysis
        - Text sentiment analysis
        - Contextual understanding
        Results: {emotion_results}

        CRITICAL CONSIDERATIONS:
        ⚠️ Pay special attention to:
        - Explicit or implicit threats
        - Signs of coercion or manipulation
        - Sudden emotional changes
        - Power dynamics in the conversation
        - Historical context of violence
        - Cultural and social factors

        DANGER SCORE CRITERIA:
        [0-59] LOW RISK
        • Normal, safe conversation
        • Positive or neutral emotions
        • No indicators of potential violence
        • No intervention needed

        [60-74] MODERATE RISK
        • Some concerning elements present
        • Potential for escalation
        • Requires monitoring but NO external notification
        • Risk level: Personal awareness needed

        [75-89] HIGH RISK
        • Significant threat indicators
        • ACTION REQUIRED: Notify registered contacts
        • System will automatically:
          - Alert trusted friends/acquaintances
          - Send detailed situation report via email
          - Request immediate attention from trusted circle
        • Risk level: Community intervention needed

        [90-100] CRITICAL RISK
        • IMMEDIATE DANGER present
        • EMERGENCY ACTION REQUIRED:
          - Automatic notification to authorities
          - Police/embassy alert system activation
          - Emergency response protocols initiated
        • Risk level: Professional intervention required

        OUTPUT REQUIREMENTS:
        1. Return ONLY a single integer between 0-100
        2. Base your score on the most severe indicators present
        3. When in doubt, err on the side of caution
        4. Consider both immediate and potential future risks

        Remember: Your assessment could save lives. Be thorough and precise.

        Return the danger score (0-100):
        """

        # Generate response
        model = genai.GenerativeModel('gemini-2.0-flash-001')
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