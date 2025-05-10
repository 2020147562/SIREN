import os
import torch
import torchaudio
import numpy as np
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Model configuration
MODEL_ID = "microsoft/wavlm-base"  # WavLM base model
EMOTION_MODEL_ID = "audeering/w2v2-emotion"  # Emotion classification model
MODEL_PATH = "/app/models"

# Emotion labels
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# 모델을 전역 변수로 한 번만 로드
logger.info("Loading emotion analysis models...")
try:
    # Load WavLM feature extractor
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID, cache_dir=MODEL_PATH)
    
    # Load emotion classification model
    model = AutoModelForAudioClassification.from_pretrained(
        EMOTION_MODEL_ID,
        cache_dir=MODEL_PATH,
        num_labels=len(EMOTION_LABELS)
    )
    
    # Set model to evaluation mode
    model.eval()
    logger.info("Models loaded successfully!")
except Exception as e:
    logger.error(f"Error loading models: {str(e)}", exc_info=True)
    raise

def analyze_audio_emotion(audio_path):
    try:
        logger.debug(f"Starting emotion analysis for audio file: {audio_path}")
        
        # Load and preprocess audio
        logger.debug(f"Loading audio file: {audio_path}")
        waveform, sample_rate = torchaudio.load(audio_path)
        logger.debug(f"Audio loaded - Shape: {waveform.shape}, Sample rate: {sample_rate}")
        
        # Validate audio length
        audio_length = waveform.shape[1] / sample_rate
        logger.debug(f"Audio length: {audio_length:.2f} seconds")
        if audio_length < 0.1:
            raise ValueError("Audio is too short (less than 0.1 seconds)")
        if audio_length > 30:
            logger.warning("Audio is longer than 30 seconds, results may not be accurate")
        
        # Convert stereo to mono if necessary
        if waveform.shape[0] > 1:
            logger.debug("Converting stereo to mono")
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            logger.debug(f"Converted waveform shape: {waveform.shape}")
        
        # Resample if necessary
        if sample_rate != 16000:
            logger.debug(f"Resampling from {sample_rate} to 16000 Hz")
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
            logger.debug(f"Resampled waveform shape: {waveform.shape}")
        
        # Preprocess audio
        logger.debug("Preprocessing audio...")
        inputs = feature_extractor(
            waveform.squeeze().numpy(),
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        logger.debug(f"Preprocessed inputs shape: {inputs.input_values.shape}")
        
        # Get model prediction using the global model
        logger.debug("Running inference...")
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
        
        # Get all emotion scores
        emotion_scores = probabilities[0].tolist()
        emotion_results = {label: score for label, score in zip(EMOTION_LABELS, emotion_scores)}
        
        # Log detailed emotion scores with confidence threshold
        logger.info("Emotion Analysis Results:")
        for emotion, score in emotion_results.items():
            if score > 0.05:  # Only show emotions with more than 5% confidence
                logger.info(f"{emotion.capitalize()}: {score:.4f} ({score*100:.2f}%)")
        
        # Get top 3 emotions
        top_emotions = sorted(emotion_results.items(), key=lambda x: x[1], reverse=True)[:3]
        logger.info("Top 3 Emotions:")
        for emotion, score in top_emotions:
            logger.info(f"{emotion.capitalize()}: {score:.4f} ({score*100:.2f}%)")
        
        predicted_emotion = EMOTION_LABELS[predicted_class]
        confidence = probabilities[0][predicted_class].item()
        
        # Add confidence threshold warning
        if confidence < 0.4:  # If confidence is less than 40%
            logger.warning(f"Low confidence prediction: {predicted_emotion.capitalize()} (Confidence: {confidence:.4f} = {confidence*100:.2f}%)")
            logger.warning("Consider using a different model or checking audio quality")
        
        logger.info(f"Predicted Emotion: {predicted_emotion.capitalize()} (Confidence: {confidence:.4f} = {confidence*100:.2f}%)")
        
        return {
            "emotion": predicted_emotion,
            "confidence": confidence,
            "all_emotions": emotion_results,
            "top_emotions": dict(top_emotions)
        }
        
    except Exception as e:
        logger.error(f"Error in emotion analysis: {str(e)}", exc_info=True)
        raise


