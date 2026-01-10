# Quality/AQ/simple_aq.py
import librosa
import numpy as np
from pathlib import Path
import json
import torch

torch.set_num_threads(1)

# 全局加载 Silero VAD 模型
VAD_MODEL = None
try:
    from silero_vad import load_silero_vad, get_speech_timestamps

    VAD_MODEL = load_silero_vad()
    print("✅ Silero VAD model loaded.")
except Exception as e:
    print("⚠️ Silero VAD not available:", e)


def compute_aq(wav_path: str):
    try:
        # 加载音频：16kHz，单声道
        y, sr = librosa.load(wav_path, sr=16000, mono=True)
        total_samples = len(y)

        # ===== 1. Clipping 检测 =====
        clip_ratio = float(np.mean(np.abs(y) > 0.99))
        if clip_ratio > 0.1:
            return {
                "aq": 0.0,
                "clip_ratio": round(clip_ratio, 4),
                "snr_db": -np.inf,
                "speech_ratio": 0.0,
            }

        # ===== 2. VAD: 计算人声占比 =====
        speech_ratio = 0.0
        speech_timestamps = []
        if VAD_MODEL is not None:
            try:
                wav_tensor = torch.from_numpy(y).float()
                speech_timestamps = get_speech_timestamps(
                    wav_tensor, VAD_MODEL, sampling_rate=16000
                )
                speech_duration = sum(
                    ts["end"] - ts["start"] for ts in speech_timestamps
                )
                speech_ratio = speech_duration / total_samples
            except Exception as e:
                print(f"VAD failed on {wav_path}: {e}")
                # 回退到 RMS 方法
                rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
                speech_ratio = float(np.mean(rms >= 0.01))
        else:
            rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
            speech_ratio = float(np.mean(rms >= 0.01))

        # 若人声占比过低，视为无效
        if speech_ratio < 0.1:
            return {
                "aq": 0.0,
                "clip_ratio": round(clip_ratio, 4),
                "snr_db": -np.inf,
                "speech_ratio": round(speech_ratio, 4),
            }

        # ===== 3. SNR 估计（基于 VAD 分段）=====
        snr_db = -np.inf
        try:
            if speech_timestamps:
                mask = np.zeros(total_samples, dtype=bool)
                for ts in speech_timestamps:
                    mask[ts["start"] : ts["end"]] = True
                speech_energy = np.mean(y[mask] ** 2) if np.any(mask) else 1e-12
                noise_energy = np.mean(y[~mask] ** 2) if np.any(~mask) else 1e-12
                snr_db = float(
                    10 * np.log10((speech_energy + 1e-12) / (noise_energy + 1e-12))
                )
            else:
                snr_db = 0.0
        except Exception:
            # 回退到能量法
            rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
            rms_db = librosa.amplitude_to_db(rms, ref=np.max)
            noise_mask = rms_db < -40
            if np.any(~noise_mask) and np.any(noise_mask):
                speech_energy = np.mean(rms[~noise_mask])
                noise_energy = np.mean(rms[noise_mask]) + 1e-12
                snr_db = float(10 * np.log10(speech_energy / noise_energy))
            else:
                snr_db = 0.0

        # ===== 4. 综合 AQ 打分 [0, 1] =====
        snr_score = min(1.0, max(0.0, snr_db / 30.0))
        aq = 0.4 * speech_ratio + 0.3 * (1.0 - clip_ratio) + 0.3 * snr_score
        aq = max(0.0, min(1.0, aq))

        return {
            "aq": round(float(aq), 4),
            "clip_ratio": round(clip_ratio, 4),
            "snr_db": round(snr_db, 2),
            "speech_ratio": round(speech_ratio, 4),
        }

    except Exception as e:
        print(f"⚠️ Error on {wav_path}: {e}")
        return {
            "aq": 0.0,
            "clip_ratio": 0.0,
            "snr_db": None,
            "speech_ratio": 0.0,
        }


def main():
    data_root = Path("/calc/users/cisri_shzh_gpu/users/lmy/asr/ASR_GT_pipline/data")
    for subset in ["dev", "train_like"]:
        results = []
        wavs_dir = data_root / subset / "wavs"
        if not wavs_dir.exists():
            print(f"⚠️ Directory not found: {wavs_dir}")
            continue

        print(f"Processing {subset}...")
        for wav in wavs_dir.glob("*.wav"):
            utt_id = wav.stem
            metrics = compute_aq(str(wav))
            result = {"utt_id": utt_id}
            result.update(metrics)
            results.append(result)

        output_file = f"{subset}_aq.jsonl"
        with open(output_file, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"✅ {subset} done ({len(results)} samples) → saved to {output_file}")


if __name__ == "__main__":
    main()
