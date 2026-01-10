# Quality/AQ/simple_aq.py
import librosa
import numpy as np
import torch

torch.set_num_threads(1)

# 全局加载 Silero VAD
VAD_MODEL = None
try:
    from silero_vad import load_silero_vad, get_speech_timestamps

    VAD_MODEL = load_silero_vad()
except Exception:
    pass


def compute_aq(wav_path: str) -> dict:
    """返回 AQ 指标字典"""
    try:
        y, sr = librosa.load(wav_path, sr=16000, mono=True)
        total_samples = len(y)

        # Clipping
        clip_ratio = float(np.mean(np.abs(y) > 0.99))
        if clip_ratio > 0.1:
            return {
                "aq": 0.0,
                "clip_ratio": round(clip_ratio, 4),
                "snr_db": -np.inf,
                "speech_ratio": 0.0,
            }

        # VAD
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
            except:
                rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
                speech_ratio = float(np.mean(rms >= 0.01))
        else:
            rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
            speech_ratio = float(np.mean(rms >= 0.01))

        if speech_ratio < 0.1:
            return {
                "aq": 0.0,
                "clip_ratio": round(clip_ratio, 4),
                "snr_db": -np.inf,
                "speech_ratio": round(speech_ratio, 4),
            }

        # SNR
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
        except:
            rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
            rms_db = librosa.amplitude_to_db(rms, ref=np.max)
            noise_mask = rms_db < -40
            if np.any(~noise_mask) and np.any(noise_mask):
                speech_energy = np.mean(rms[~noise_mask])
                noise_energy = np.mean(rms[noise_mask]) + 1e-12
                snr_db = float(10 * np.log10(speech_energy / noise_energy))
            else:
                snr_db = 0.0

        # AQ 打分
        snr_score = min(1.0, max(0.0, snr_db / 30.0))
        aq = 0.5 * speech_ratio + 0.3 * (1.0 - clip_ratio) + 0.3 * snr_score
        aq = max(0.0, min(1.0, aq))

        return {
            "aq": round(float(aq), 4),
            "clip_ratio": round(clip_ratio, 4),
            "snr_db": round(snr_db, 2),
            "speech_ratio": round(speech_ratio, 4),
        }

    except Exception as e:
        return {"aq": 0.0, "clip_ratio": 0.0, "snr_db": None, "speech_ratio": 0.0}
