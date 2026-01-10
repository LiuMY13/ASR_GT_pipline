# merge_full_features.py
import json
from pathlib import Path
import sys

BASE_DIR = Path("/calc/users/cisri_shzh_gpu/users/lmy/asr/ASR_GT_pipline")
sys.path.insert(0, str(BASE_DIR / "Quality/AQ"))
sys.path.insert(0, str(BASE_DIR / "Quality/TQ"))

from aq import compute_aq
from tq import compute_tq
from run_funasr_eval import run_teacher_asr

OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def main():
    dev_dir = BASE_DIR / "data" / "dev_1"
    MODEL_PATH = str(BASE_DIR / "FunAudioLLM/Fun-ASR-Nano-2512")
    LM_MODEL_PATH = str(BASE_DIR / "Quality/TQ/uer/gpt2-chinese-cluecorpussmall")

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

        # Paths
        wav_path = str(dev_dir / meta_item["audio_path"])
        text_online = meta_item["text_online"]
        text_gt = gt_map[utt_id]

        # === AQ ===
        aq_metrics = compute_aq(wav_path)

        # === TQ (online) ===
        tq_online = compute_tq(text_online)

        # === Teacher result ===
        teacher_res = teacher_results.get(utt_id, {})
        text_teacher = teacher_res.get("hyp_fun_asr", "")
        teacher_cer = teacher_res.get("cer", None)
        teacher_wer = teacher_res.get("wer", None)

        # === TQ (teacher) ===
        tq_teacher = compute_tq(text_teacher)

        # === TQ (GT) ===
        tq_gt = compute_tq(text_gt)

        # === Final record ===
        record = {
            "utt_id": utt_id,
            "audio_path": meta_item["audio_path"],
            "duration_sec": meta_item.get("duration_sec", None),
            # Texts
            "text_gt": text_gt,
            "text_online": text_online,
            "text_teacher": text_teacher,
            # AQ Metrics
            **aq_metrics,
            # Teacher ASR Metrics
            "teacher_cer": teacher_cer,
            "teacher_wer": teacher_wer,
            # TQ Metrics
            "hyp_online_lm_logprob": tq_online["lm_logprob"],
            "hyp_teacher_lm_logprob": tq_teacher["lm_logprob"],
            "hyp_online_tq": tq_online["tq"],
            "hyp_teacher_tq": tq_teacher["tq"],
            "ref_lm_logprob": tq_gt["lm_logprob"],
            "ref_tq": tq_gt["tq"],
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
