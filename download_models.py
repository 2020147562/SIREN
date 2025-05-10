import os
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor, AutoModelForSequenceClassification
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_models():
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Download audio emotion recognition model
    logger.info("Downloading audio emotion recognition model...")
    AutoModelForAudioClassification.from_pretrained(
        "superb/wav2vec2-large-superb-er",
        cache_dir="/app/models"
    )
    logger.info("Audio model downloaded successfully.")

    # Download text emotion recognition model
    logger.info("Downloading text emotion recognition model...")
    AutoModelForSequenceClassification.from_pretrained(
        "j-hartmann/emotion-english-distilroberta-base",
        cache_dir="/app/models"
    )
    logger.info("Text model downloaded successfully.")

if __name__ == "__main__":
    download_models() 