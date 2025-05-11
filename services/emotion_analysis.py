import torch
import torchaudio
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

# -----------------------------
# 1. Load audio model & extractor
# -----------------------------
logger.info("Loading audio model and feature extractor...")
feature_extractor = AutoFeatureExtractor.from_pretrained(AUDIO_MODEL_ID)
audio_model = Wav2Vec2ForSequenceClassification.from_pretrained(
    AUDIO_MODEL_ID,
    ignore_mismatched_sizes=True
).to(device)
audio_model.eval()
logger.info("Audio model loaded.")

# -----------------------------
# 2. Load text model & tokenizer
# -----------------------------
logger.info("Loading text tokenizer and model...")
text_tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_ID)
text_model = AutoModelForSequenceClassification.from_pretrained(TEXT_MODEL_ID).to(device)
text_model.eval()
logger.info("Text model loaded.")

# -----------------------------
# 3. Original labels for reference
# -----------------------------
# Audio model labels:
#   "ang", "hap", "sad", "neu"
#
# Text model labels:
#   "anger", "disgust", "fear", "joy",
#   "neutral", "sadness", "surprise"

# -----------------------------
# 4. Label mapping for similar labels
# -----------------------------
LABEL_MAPPING = {
    # audio → text 라벨로 매핑
    "ang":     "anger",
    "hap":     "joy",
    "neu":     "neutral",
    "sad":     "sadness",
    # (텍스트 모델 라벨은 그대로 사용되므로 별도 매핑 불필요)
}

def map_labels(scores: dict) -> dict:
    """
    Apply LABEL_MAPPING to merge similar labels, summing their scores.
    """
    mapped = {}
    for label, score in scores.items():
        new_label = LABEL_MAPPING.get(label, label)
        mapped[new_label] = mapped.get(new_label, 0.0) + score
    return mapped

def fuse_union(audio_scores: dict, text_scores: dict) -> dict:
    """
    Union fusion:
      - map labels first
      - for labels in both: average
      - for labels in only one: take that score
    """
    a_map = map_labels(audio_scores)
    t_map = map_labels(text_scores)
    all_labels = sorted(set(a_map) | set(t_map))
    fused = {}
    for lbl in all_labels:
        a = a_map.get(lbl)
        t = t_map.get(lbl)
        if a is not None and t is not None:
            fused[lbl] = (a + t) / 2
        else:
            fused[lbl] = a if t is None else t
    return fused

# -----------------------------
# 5. Audio preprocessing utils
# -----------------------------
@lru_cache(maxsize=None)
def get_resampler(orig_sr: int):
    return torchaudio.transforms.Resample(orig_sr, feature_extractor.sampling_rate)

WINDOW_SIZE = 5.0  # seconds
OVERLAP = 1.0      # seconds

def analyze_audio(audio_path: str) -> dict:
    waveform, sr = torchaudio.load(audio_path)
    # to mono
    if waveform.dim() > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    # resample if needed
    if sr != feature_extractor.sampling_rate:
        waveform = get_resampler(sr)(waveform)
        sr = feature_extractor.sampling_rate

    total_sec = waveform.shape[1] / sr
    step = WINDOW_SIZE - OVERLAP
    segments = []
    start = 0.0
    # slice into overlapping windows
    while start < total_sec:
        end = min(start + WINDOW_SIZE, total_sec)
        seg = waveform[:, int(start * sr):int(end * sr)].squeeze(0)  # 1D
        segments.append(seg)
        start += step

    # Convert list of 1D torch.Tensors to numpy arrays for feature_extractor
    segments_np = [seg.cpu().numpy() for seg in segments]

    # Let the feature_extractor handle padding and batching
    inputs = feature_extractor(
        segments_np,
        sampling_rate=sr,
        return_tensors="pt",
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = audio_model(**inputs).logits  # (batch_size, num_labels)
        probs = torch.nn.functional.softmax(logits, dim=-1).mean(dim=0)

    labels = [audio_model.config.id2label[i] for i in range(probs.size(0))]
    audio_scores = {labels[i]: float(probs[i]) for i in range(len(labels))}
    return audio_scores

def analyze_text(text: str) -> dict:
    inputs = text_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = text_model(**inputs).logits  # (1, num_labels)
        probs = torch.nn.functional.softmax(logits, dim=-1)[0]

    labels = [text_model.config.id2label[i] for i in range(probs.size(0))]
    text_scores = {labels[i]: float(probs[i]) for i in range(len(labels))}
    return text_scores

# -----------------------------
# 6. Final emotion analysis with logging
# -----------------------------
def analyze_emotion(audio_path: str, text: str) -> dict:
    """
    Run audio & text analysis in parallel, then fuse via union heuristic.
    Logs intermediate and fused results.
    Returns: {
        "emotion":    str,               # top label
        "confidence": float,             # its fused score
        "all_emotions": dict[label:score]
    }
    """
    with ThreadPoolExecutor() as executor:
        fa = executor.submit(analyze_audio, audio_path) if audio_path else None
        ft = executor.submit(analyze_text, text)
        audio_scores = fa.result() if fa else {}
        text_scores = ft.result()

    # 1) Log audio model results
    logger.debug(f"Audio model scores: {audio_scores}")

    # 2) Log text model results
    logger.debug(f"Text model scores: {text_scores}")

    # 3) Fuse and log fused results
    fused = fuse_union(audio_scores, text_scores)
    logger.debug(f"Fused emotion scores: {fused}")

    top = max(fused, key=fused.get)
    return {
        "emotion":      top,
        "confidence":   fused[top],
        "all_emotions": fused
    }
