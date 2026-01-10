# Quality/AQ/simple_aq.py
import librosa
import numpy as np
from pathlib import Path
import json


def compute_aq(wav_path: str) -> float:
    try:
        y, sr = librosa.load(wav_path, sr=16000)
        duration = len(y) / sr

        # 1. 时长过滤
        if duration < 0.5 or duration > 30.0:
            return 0.0

        # 2. 静音检测 (RMS 能量)
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        silence_ratio = np.mean(rms < 0.01)
        if silence_ratio > 0.8:
            return 0.0

        # 3. Clipping 检测
        clip_ratio = np.mean(np.abs(y) > 0.99)
        if clip_ratio > 0.1:
            return 0.0

        # 综合打分 [0, 1]
        aq = 0.6 * (1.0 - silence_ratio) + 0.4 * (1.0 - clip_ratio)
        return max(0.0, min(1.0, aq))
    except Exception as e:
        print(f"⚠️ Error on {wav_path}: {e}")
        return 0.0


def main():
    data_root = Path("/calc/users/cisri_shzh_gpu/users/lmy/asr/ASR_GT_pipline/data")
    for subset in ["dev", "train_like"]:
        results = []
        for wav in (data_root / subset / "wavs").glob("*.wav"):
            utt_id = wav.stem
            aq = compute_aq(str(wav))
            results.append({"utt_id": utt_id, "aq": round(aq, 4)})

        with open(f"{subset}_aq.jsonl", "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"✅ {subset} done ({len(results)} samples)")


if __name__ == "__main__":
    main()
