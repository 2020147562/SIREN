import os
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_models():
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    logger.info("Downloading emotion recognition model...")
    AutoModelForAudioClassification.from_pretrained(
        "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
        cache_dir="/app/models"
    )
    logger.info("Model downloaded successfully.")

if __name__ == "__main__":
    download_models() 