# loglabel_booster/run.py (enhanced for dev evaluation)
import json
import argparse
from pathlib import Path
import sys

# BASE_DIR = Path("/calc/users/cisri_shzh_gpu/users/lmy/asr/ASR_GT_pipline")
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "Quality"))

from AQ.aq import compute_aq
from TQ.tq import compute_tq
from run_funasr_eval import run_teacher_asr
from utils.text_norm import text_normalize
from jiwer import cer, wer
import jieba


def tokenize_for_wer(text: str) -> str:
    words = jieba.lcut(text)
    return " ".join([w.strip() for w in words if w.strip()])


def decide_final_label(
    text_online: str, text_teacher: str, aq: float, tq_online: float, tq_teacher: float
) -> tuple[str, bool, list[str]]:
    hyp_online = text_normalize(text_online)
    hyp_teacher = text_normalize(text_teacher)
    agreement_cer = cer(hyp_online, hyp_teacher)

    if aq < 0.4:
        return "", False, ["low_aq"]
    if tq_online >= 0.6:
        return text_online, True, ["high_online_tq"]
    if tq_teacher >= 0.6 and tq_online < 0.6 and agreement_cer > 0.2:
        return text_teacher, True, ["teacher_replace", "teacher_better"]
    return text_online, True, ["default_keep"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--subset", type=str, choices=["dev", "train_like", "test"], required=True
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    data_dir = input_dir / args.subset
    MODEL_PATH = str(BASE_DIR / "FunAudioLLM/Fun-ASR-Nano-2512")

    # === 1. 加载 meta.jsonl ===
    meta_list = []
    with open(data_dir / "meta.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                meta_list.append(json.loads(line))

    # === 2. 仅 dev 需要加载 GT ===
    gt_map = {}
    if args.subset == "dev":
        with open(data_dir / "gt.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    gt_map[item["utt_id"]] = item["text_gt"]

    # === 3. 运行 Teacher ASR ===
    print(f"🚀 Running Teacher ASR on {args.subset}...")
    # 修改 run_teacher_asr 以支持无 GT 模式（见下方说明）
    teacher_results = run_teacher_asr(data_dir, MODEL_PATH)

    # === 4. 处理每条样本 ===
    manifest = []
    per_utt_metrics = []  # 仅用于 dev

    for meta_item in meta_list:
        utt_id = meta_item["utt_id"]
        wav_path = str(data_dir / meta_item["audio_path"])
        text_online = meta_item["text_online"]

        # AQ & TQ
        aq = compute_aq(wav_path)["aq"]
        tq_online = compute_tq(text_online)["tq"]
        text_teacher = teacher_results.get(utt_id, {}).get("hyp_fun_asr", "")
        tq_teacher = compute_tq(text_teacher)["tq"]

        # 决策
        text_final, keep, tags = decide_final_label(
            text_online, text_teacher, aq, tq_online, tq_teacher
        )

        # 构建 manifest
        record = {
            "utt_id": utt_id,
            "audio_path": meta_item["audio_path"],
            "text_online": text_online,
            "text_teacher": text_teacher,
            "text_final": text_final,
            "keep": keep,
            "aq": float(aq),
            "tq": (
                float(tq_online)
                if "high_online_tq" in tags or "default_keep" in tags
                else float(tq_teacher)
            ),
            "tags": tags,
        }
        manifest.append(record)

        # === 5. 仅 dev 计算 per-utt CER/WER ===
        if args.subset == "dev" and utt_id in gt_map:
            ref_raw = gt_map[utt_id]
            ref_cer = text_normalize(ref_raw)
            ref_wer = tokenize_for_wer(ref_cer)

            # Baseline (online)
            hyp_online_cer = text_normalize(text_online)
            hyp_online_wer = tokenize_for_wer(hyp_online_cer)
            online_cer = cer(ref_cer, hyp_online_cer)
            online_wer = wer(ref_wer, hyp_online_wer)

            # Final result
            if keep:
                hyp_final_cer = text_normalize(text_final)
                hyp_final_wer = tokenize_for_wer(hyp_final_cer)
                final_cer = cer(ref_cer, hyp_final_cer)
                final_wer = wer(ref_wer, hyp_final_wer)
            else:
                final_cer = 1.0
                final_wer = 1.0

            per_utt_metrics.append(
                {
                    "utt_id": utt_id,
                    "online_cer": round(online_cer, 4),
                    "online_wer": round(online_wer, 4),
                    "final_cer": round(final_cer, 4),
                    "final_wer": round(final_wer, 4),
                    "duration_sec": meta_item.get("duration_sec", 1.0),
                    "keep": keep,
                }
            )

    # === 6. 保存 manifest ===
    with open(output_dir / "manifest.jsonl", "w", encoding="utf-8") as f:
        for rec in manifest:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # === 7. 仅 dev 保存指标 ===
    if args.subset == "dev":
        # 计算加权指标
        total_duration = sum(m["duration_sec"] for m in per_utt_metrics)
        weighted_online_cer = (
            sum(m["online_cer"] * m["duration_sec"] for m in per_utt_metrics)
            / total_duration
        )
        weighted_final_cer = (
            sum(m["final_cer"] * m["duration_sec"] for m in per_utt_metrics)
            / total_duration
        )
        weighted_online_wer = (
            sum(m["online_wer"] * m["duration_sec"] for m in per_utt_metrics)
            / total_duration
        )
        weighted_final_wer = (
            sum(m["final_wer"] * m["duration_sec"] for m in per_utt_metrics)
            / total_duration
        )

        dev_score = {
            "baseline_weighted_cer": round(weighted_online_cer, 4),
            "final_weighted_cer": round(weighted_final_cer, 4),
            "cer_improvement": round(weighted_online_cer - weighted_final_cer, 4),
            "baseline_weighted_wer": round(weighted_online_wer, 4),
            "final_weighted_wer": round(weighted_final_wer, 4),
            "wer_improvement": round(weighted_online_wer - weighted_final_wer, 4),
            "coverage": round(
                sum(1 for m in per_utt_metrics if m["keep"]) / len(per_utt_metrics), 4
            ),
            "num_samples": len(per_utt_metrics),
        }

        # 保存 per-utt 和 summary
        with open(output_dir / "per_utt_metrics.jsonl", "w", encoding="utf-8") as f:
            for m in per_utt_metrics:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

        with open(output_dir / "dev_score.json", "w", encoding="utf-8") as f:
            json.dump(dev_score, f, ensure_ascii=False, indent=2)

        print(f"📊 Dev Metrics:")
        print(
            f"  Baseline CER: {weighted_online_cer:.4f} → Final CER: {weighted_final_cer:.4f}"
        )
        print(f"  Coverage: {dev_score['coverage']:.2%}")

    print(f"✅ Done! Processed {len(manifest)} samples.")
    print(f"📄 Output saved to {output_dir}/")


if __name__ == "__main__":
    main()


"""
# 处理 dev 集
python run.py \
  --input_dir data/ \
  --output_dir outputs/dev/ \
  --subset dev

# 处理 train_like（用于训练）
python run.py \
  --input_dir data/ \
  --output_dir outputs/train_like/ \
  --subset train_like

# 处理 test（用于提交）
python run.py \
  --input_dir interview_data/ \
  --output_dir outputs/test/ \
  --subset test
  
"""
# python run_test.py \
#   --input_dir data/ \
#   --output_dir outputs/dev_test/ \
#   --subset dev
