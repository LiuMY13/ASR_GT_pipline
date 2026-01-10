import json
from pathlib import Path
from jiwer import cer, wer
from utils.text_norm import text_normalize
import jieba


def tokenize_for_wer(text: str) -> str:
    """Tokenize text for WER using jieba (handles Chinese + English)."""
    words = jieba.lcut(text)
    words = [word.strip() for word in words if word.strip()]
    return " ".join(words)


def compute_per_utt_cer_wer(ref: str, hyp: str) -> tuple[float, float]:
    """Compute CER and WER for a single utterance."""
    # print(f"[DEBUG] ref={repr(ref)} | hyp={repr(hyp)}")
    try:
        c = cer(ref, hyp)
    except Exception:
        c = 1.0  # fallback
    try:
        w = wer(ref, hyp)
    except Exception:
        w = 1.0  # fallback
    return c, w


def main():
    dev_dir = Path("data/dev")
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    # 1. Load GT
    gt_map = {}
    with open(dev_dir / "gt.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                gt_map[item["utt_id"]] = item["text_gt"]

    # 2. Prepare lists for overall metrics
    refs_cer_all = []
    hyps_cer_all = []
    refs_wer_all = []
    hyps_wer_all = []

    # 3. Per-utterance metrics list
    per_utt_list = []

    with open(dev_dir / "meta.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            utt_id = item["utt_id"]
            if utt_id not in gt_map:
                continue

            ref_raw = gt_map[utt_id]
            hyp_raw = item["text_online"]

            # Normalize for CER
            ref_cer = text_normalize(ref_raw)
            hyp_cer = text_normalize(hyp_raw)

            if not ref_cer:
                continue

            # Tokenize for WER
            ref_wer = tokenize_for_wer(ref_cer)
            hyp_wer = tokenize_for_wer(hyp_cer)

            # Compute per-utt CER/WER
            utt_cer, _ = compute_per_utt_cer_wer(ref_cer, hyp_cer)
            _, utt_wer = compute_per_utt_cer_wer(ref_wer, hyp_wer)

            # Save for overall metrics
            refs_cer_all.append(ref_cer)
            hyps_cer_all.append(hyp_cer)
            refs_wer_all.append(ref_wer)
            hyps_wer_all.append(hyp_wer)

            # Save per-utt record
            per_utt_list.append(
                {
                    "utt_id": utt_id,
                    "ref_raw": ref_raw,
                    "hyp_online": hyp_raw,
                    "ref_cer": ref_cer,
                    "hyp_cer": hyp_cer,
                    "ref_wer": ref_wer,  # ← 新增
                    "hyp_wer": hyp_wer,
                    "cer": round(utt_cer, 4),
                    "wer": round(utt_wer, 4),
                }
            )

    # 4. Compute overall metrics
    overall_cer = cer(refs_cer_all, hyps_cer_all)
    overall_wer = wer(refs_wer_all, hyps_wer_all)

    # 5. Save overall score
    overall_result = {
        "baseline_cer": round(overall_cer, 4),
        "baseline_wer": round(overall_wer, 4),
        "num_samples": len(per_utt_list),
        "normalize_rules": "NFKC + lower + remove punctuation + collapse whitespace",
        "wer_tokenization": "jieba for Chinese",
    }

    with open(output_dir / "dev_score.json", "w", encoding="utf-8") as f:
        json.dump(overall_result, f, ensure_ascii=False, indent=2)

    # 6. Save per-utterance metrics
    with open(output_dir / "per_utt_metrics.jsonl", "w", encoding="utf-8") as f:
        for record in per_utt_list:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✅ Overall CER: {overall_cer:.4f}")
    print(f"✅ Overall WER: {overall_wer:.4f}")
    print(f"📊 ({len(per_utt_list)} samples)")
    print(f"📝 Overall score saved to outputs/dev_score.json")
    print(f"📄 Per-utterance metrics saved to outputs/per_utt_metrics.jsonl")


if __name__ == "__main__":
    main()
