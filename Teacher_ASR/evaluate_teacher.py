# Teacher_ASR/evaluate_teacher.py
import json
from pathlib import Path
from jiwer import cer, wer
import sys

sys.path.append(str(Path(__file__).parent.parent))  # 添加项目根目录到 path
from utils.text_norm import text_normalize
from Teacher_ASR.asr_model import FasterWhisperASR
from utils.text_norm import text_normalize
import jieba


def tokenize_for_wer(text: str) -> str:
    words = jieba.lcut(text)
    words = [w.strip() for w in words if w.strip()]
    return " ".join(words)


def main():
    output_dir = Path("outputs")
    dev_gt_path = Path("../data/dev/gt.jsonl")
    dev_teacher_path = output_dir / "dev_with_teacher.jsonl"

    # Load GT
    gt_map = {}
    with open(dev_gt_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                gt_map[item["utt_id"]] = item["text_gt"]

    # Prepare lists
    refs_cer, hyps_online, hyps_teacher = [], [], []
    refs_wer, hyps_online_wer, hyps_teacher_wer = [], [], []

    with open(dev_teacher_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            utt_id = item["utt_id"]
            if utt_id not in gt_map:
                continue

            ref_raw = gt_map[utt_id]
            hyp_online_raw = item["text_online"]
            hyp_teacher_raw = item["text_teacher"]

            # Normalize for CER
            ref_cer = text_normalize(ref_raw)
            hyp_online_cer = text_normalize(hyp_online_raw)
            hyp_teacher_cer = hyp_teacher_raw  # already normalized

            if not ref_cer:
                continue

            # Tokenize for WER
            ref_wer = tokenize_for_wer(ref_cer)
            hyp_online_wer = tokenize_for_wer(hyp_online_cer)
            hyp_teacher_wer = tokenize_for_wer(hyp_teacher_cer)

            # Append
            refs_cer.append(ref_cer)
            hyps_online.append(hyp_online_cer)
            hyps_teacher.append(hyp_teacher_cer)

            refs_wer.append(ref_wer)
            hyps_online_wer.append(hyp_online_wer)
            hyps_teacher_wer.append(hyp_teacher_wer)

    # Compute metrics
    cer_online = cer(refs_cer, hyps_online)
    cer_teacher = cer(refs_cer, hyps_teacher)
    wer_online = wer(refs_wer, hyps_online_wer)
    wer_teacher = wer(refs_wer, hyps_teacher_wer)

    # Save results
    result = {
        "baseline_cer": round(cer_online, 4),
        "teacher_cer": round(cer_teacher, 4),
        "baseline_wer": round(wer_online, 4),
        "teacher_wer": round(wer_teacher, 4),
        "cer_improvement": round(cer_online - cer_teacher, 4),
        "wer_improvement": round(wer_online - wer_teacher, 4),
        "num_samples": len(refs_cer),
    }

    with open(output_dir / "teacher_score.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("📊 Teacher ASR Evaluation:")
    print(
        f"  Baseline CER: {cer_online:.4f} → Teacher CER: {cer_teacher:.4f} (Δ={cer_online - cer_teacher:.4f})"
    )
    print(
        f"  Baseline WER: {wer_online:.4f} → Teacher WER: {wer_teacher:.4f} (Δ={wer_online - wer_teacher:.4f})"
    )
    print(f"📝 Results saved to {output_dir}/teacher_score.json")


if __name__ == "__main__":
    main()
