import google.generativeai as genai
import os
import logging
# from rag.rag_engine import search_similar_cases

logger = logging.getLogger(__name__)

# Configure the Gemini API
def configure_gemini():
    try:
        # Get API key from environment variable
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set")
        
        # Configure the API
        genai.configure(api_key=api_key)
        
        # Get available models
        models = genai.list_models()
        logger.info("Available Gemini models:")
        for model in models:
            logger.info(f"- {model.name}")
            logger.info(f"  Supported methods: {model.supported_generation_methods}")
        
        return True
    except Exception as e:
        logger.error(f"Error configuring Gemini API: {str(e)}")
        return False

# Initialize Gemini configuration
if not configure_gemini():
    raise RuntimeError("Failed to configure Gemini API")

def analyze_text_and_decide(text: str, emotion_result: dict) -> tuple[float, str]:
    try:
        # Create the model instance - using the flash model for better quota management
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Create the prompt with emotion analysis results
        emotion_str = "\n".join([f"{emotion.capitalize()}: {score:.2%}" for emotion, score in emotion_result.items()])
        prompt = f"""
        Analyze the following text and emotion analysis results to determine the level of risk.
        
        Text:
        {text}
        
        Emotion Analysis Results:
        {emotion_str}
        
        Return ONLY a single number between 0-100 representing the risk level, where:
        - 0-59: Safe or minor concerns
        - 60-74: Moderate risk - potential for minor physical/mental violence, but not reportable
        - 75-89: High risk - reportable to acquaintances, potential for moderate physical/mental violence
        - 90-100: Severe risk - reportable to authorities (police/embassy), significant physical/mental violence
        
        Do not include any other text or explanation, just the number.
        """
        
        # Generate response using generateContent
        response = model.generate_content(prompt)
        
        # Extract danger score from response
        danger_score = 0.0  # Default to safe
        try:
            # Extract the first number from the response
            import re
            numbers = re.findall(r'\b(?:100|[1-9]?[0-9])\b', response.text)
            if numbers:
                danger_score = float(numbers[0])
                logger.info(f"Extracted danger score: {danger_score}")
            else:
                logger.warning("No valid score found in response")
        except Exception as e:
            logger.warning(f"Could not extract danger score from response: {str(e)}")
        
        return danger_score, response.text
        
    except Exception as e:
        logger.error(f"Error in analyze_text_and_decide: {str(e)}")
        raise
