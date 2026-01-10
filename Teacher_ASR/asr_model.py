# Teacher_ASR/asr_model.py
from faster_whisper import WhisperModel
import torch
from pathlib import Path


class FasterWhisperASR:
    def __init__(self, model_size="medium", device="auto", compute_type="float16"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # 本地部署
        model_path = Path(__file__).parent / "faster-whisper-medium"

        # 从hf下载
        # self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

        # 本地部署
        self.model = WhisperModel(
            str(model_path),  # ← 关键：传入本地路径（字符串）
            device=device,
            compute_type=compute_type,
        )

    def transcribe(self, audio_path: str) -> str:
        segments, _ = self.model.transcribe(
            audio_path,
            beam_size=5,
            # language="zh",  # 强制中文（可提升中文准确率）
            language=None,  # 因为我们的语料中英混杂
            # without_punctuation=True,  # 移除标点（需 faster-whisper >= 1.0）
            word_timestamps=False,
        )
        text = "".join([seg.text for seg in segments])
        return text.strip()
