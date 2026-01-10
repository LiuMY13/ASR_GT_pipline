# Teacher_ASR/asr_model.py
from faster_whisper import WhisperModel
import torch


class FasterWhisperASR:
    def __init__(self, model_size="medium", device="auto", compute_type="float16"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading faster-whisper ({model_size}) on {device}...")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio_path: str) -> str:
        segments, _ = self.model.transcribe(
            audio_path,
            beam_size=5,
            # language="zh",  # 强制中文（可提升中文准确率）
            language=None,  # 因为我们的预料中英混杂
            without_punctuation=True,  # 移除标点（需 faster-whisper >= 1.0）
            word_timestamps=False,
        )
        text = "".join([seg.text for seg in segments])
        return text.strip()
