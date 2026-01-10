# merge_full_features.py
import json
from pathlib import Path
import sys

BASE_DIR = Path("/calc/users/cisri_shzh_gpu/users/lmy/asr/ASR_GT_pipline")
sys.path.insert(0, str(BASE_DIR / "Quality/AQ"))
sys.path.insert(0, str(BASE_DIR / "Quality/TQ"))

# ===== 复用你的文本处理工具 =====
try:
    from utils.text_norm import text_normalize
except ImportError:
    # fallback normalize (与你的 eval_baseline.py 一致)
    import unicodedata
    import re

    def text_normalize(text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = unicodedata.normalize("NFKC", text).lower()
        text = re.sub(r"[^\u4e00-\u9fa5a-z0-9]", " ", text)
        return re.sub(r"\s+", " ", text).strip()


import jieba


def tokenize_for_wer(text: str) -> str:
    words = jieba.lcut(text)
    return " ".join([w.strip() for w in words if w.strip()])


from jiwer import cer, wer


def compute_online_cer_wer(ref: str, hyp: str) -> tuple[float, float]:
    try:
        c = cer(ref, hyp)
    except:
        c = 1.0
    try:
        w = wer(ref, hyp)
    except:
        w = 1.0
    return c, w


# ===== 其他导入 =====
from aq import compute_aq
from tq import compute_tq
from run_funasr_eval import run_teacher_asr

OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def main():
    dev_dir = BASE_DIR / "data" / "dev"  # 注意：你之前写的是 dev_1，这里改为 dev
    MODEL_PATH = str(BASE_DIR / "FunAudioLLM/Fun-ASR-Nano-2512")

    # === 1. Load GT and meta ===
    gt_map = {}
    with open(dev_dir / "gt.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                gt_map[item["utt_id"]] = item["text_gt"]

    meta_list = []
    with open(dev_dir / "meta.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                meta_list.append(json.loads(line))

    # === 2. Run Teacher ASR ===
    print("🚀 Running Teacher ASR...")
    teacher_results = run_teacher_asr(dev_dir, MODEL_PATH)

    # === 3. Process each utterance ===
    full_list = []
    for meta_item in meta_list:
        utt_id = meta_item["utt_id"]
        if utt_id not in gt_map:
            continue

        wav_path = str(dev_dir / meta_item["audio_path"])
        text_online = meta_item["text_online"]
        text_gt = gt_map[utt_id]

        # === 计算 online_cer / online_wer ===
        ref_cer = text_normalize(text_gt)
        hyp_cer = text_normalize(text_online)
        if not ref_cer:
            continue

        ref_wer = tokenize_for_wer(ref_cer)
        hyp_wer = tokenize_for_wer(hyp_cer)

        online_cer, _ = compute_online_cer_wer(ref_cer, hyp_cer)
        _, online_wer = compute_online_cer_wer(ref_wer, hyp_wer)

        # === AQ ===
        aq_metrics = compute_aq(wav_path)

        # === TQ ===
        tq_online = compute_tq(text_online)
        tq_teacher = compute_tq(teacher_results.get(utt_id, {}).get("hyp_fun_asr", ""))
        tq_gt = compute_tq(text_gt)

        # === Final record ===
        record = {
            "utt_id": utt_id,
            "audio_path": meta_item["audio_path"],
            "duration_sec": meta_item.get("duration_sec", None),
            # Texts
            "text_gt": text_gt,
            "text_online": text_online,
            "text_teacher": teacher_results.get(utt_id, {}).get("hyp_fun_asr", ""),
            # Online vs GT Metrics
            "online_cer": round(online_cer, 4),
            "online_wer": round(online_wer, 4),
            # AQ Metrics
            **aq_metrics,
            # Teacher ASR Metrics
            "teacher_cer": teacher_results.get(utt_id, {}).get("cer", None),
            "teacher_wer": teacher_results.get(utt_id, {}).get("wer", None),
            # TQ Metrics
            "ref_lm_logprob": tq_gt["lm_logprob"],
            "ref_tq": tq_gt["tq"],
            "hyp_online_lm_logprob": tq_online["lm_logprob"],
            "hyp_teacher_lm_logprob": tq_teacher["lm_logprob"],
            "hyp_online_tq": tq_online["tq"],
            "hyp_teacher_tq": tq_teacher["tq"],
        }

        full_list.append(record)

    # === Save ===
    output_file = OUTPUT_DIR / "dev_full_features.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for rec in full_list:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"✅ Done! Merged {len(full_list)} samples to {output_file}")


if __name__ == "__main__":
    main()
