import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoFeatureExtractor,
    Wav2Vec2ForSequenceClassification,
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model configuration
AUDIO_MODEL_ID = "superb/wav2vec2-large-superb-er"
TEXT_MODEL_ID = "j-hartmann/emotion-english-distilroberta-base"

# Load audio feature extractor and model
logger.info("Loading audio model and feature extractor...")
feature_extractor = AutoFeatureExtractor.from_pretrained(AUDIO_MODEL_ID)
audio_model = Wav2Vec2ForSequenceClassification.from_pretrained(
    AUDIO_MODEL_ID,
    ignore_mismatched_sizes=True
).to(device)
audio_model.eval()
logger.info("Audio model loaded.")

# Load text tokenizer and model
logger.info("Loading text tokenizer and model...")
text_tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_ID)
text_model = AutoModelForSequenceClassification.from_pretrained(TEXT_MODEL_ID).to(device)
text_model.eval()
logger.info("Text model loaded.")

# Determine common emotion labels
audio_labels = set(audio_model.config.id2label.values())
text_labels = set(text_model.config.id2label.values())
EMOTION_LABELS = sorted(audio_labels & text_labels)

# Cache resampler
@lru_cache(maxsize=None)
def get_resampler(orig_sr):
    return torchaudio.transforms.Resample(orig_sr, feature_extractor.sampling_rate)

# Sliding window parameters
WINDOW_SIZE = 5.0    # seconds
OVERLAP = 1.0        # seconds


def analyze_audio(audio_path: str) -> dict:
    waveform, sr = torchaudio.load(audio_path)
    # mono
    if waveform.dim() > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    # resample
    if sr != feature_extractor.sampling_rate:
        waveform = get_resampler(sr)(waveform)
        sr = feature_extractor.sampling_rate

    total_sec = waveform.shape[1] / sr
    step = WINDOW_SIZE - OVERLAP
    segments = []
    start = 0.0
    # 1) 세그먼트 자르고
    while start < total_sec:
        end = min(start + WINDOW_SIZE, total_sec)
        seg = waveform[:, int(start * sr):int(end * sr)].squeeze(0)
        segments.append(seg)
        start += step

    # 2) 길이가 다른 세그먼트를 패딩해서 (batch, time) 텐서로 만들기
    # pad_sequence는 [T_i] 리스트를 [batch, max(T_i)] 로 채워줍니다.
    segments_padded = pad_sequence(segments, batch_first=True)  # shape: (batch, max_len)

    # 3) feature_extractor에 바로 넘기기
    #    huggingface feature_extractor는 torch.Tensor 도 받아줍니다.
    inputs = feature_extractor(
        segments_padded,           # torch.Tensor
        sampling_rate=sr,
        return_tensors="pt",
        padding=True               # 사실 이 옵션은 이제 필요 없지만 남겨도 안전합니다
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = audio_model(**inputs).logits    # (batch, num_labels)
        probs = torch.nn.functional.softmax(logits, dim=-1).mean(dim=0)

    return {
        label: float(probs[audio_model.config.label2id[label]])
        for label in EMOTION_LABELS
    }


def analyze_text(text: str) -> dict:
    inputs = text_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = text_model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=-1)[0]

    return {label: float(probs[text_model.config.label2id[label]]) for label in EMOTION_LABELS}


def fuse_emotions(audio_scores: dict, text_scores: dict, alpha: float = 0.5) -> dict:
    return {label: alpha * audio_scores.get(label, 0.0) + (1 - alpha) * text_scores.get(label, 0.0)
            for label in EMOTION_LABELS}


def analyze_emotion(audio_path: str, text: str) -> dict:
    """
    Parallel analysis of audio and text, then fuse scores.
    """
    with ThreadPoolExecutor() as executor:
        future_audio = executor.submit(analyze_audio, audio_path) if audio_path else None
        future_text = executor.submit(analyze_text, text)
        audio_scores = future_audio.result() if future_audio else {label: 0.0 for label in EMOTION_LABELS}
        text_scores = future_text.result()

    fused = fuse_emotions(audio_scores, text_scores)
    top_emotion = max(fused, key=fused.get)
    return {"emotion": top_emotion, "confidence": fused[top_emotion], "all_emotions": fused}