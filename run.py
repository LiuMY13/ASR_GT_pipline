# loglabel_booster/run.py
import json
import argparse
from pathlib import Path
import sys

# 添加模块路径
BASE_DIR = Path(__file__).parent.parent

BASE_DIR = Path("/calc/users/cisri_shzh_gpu/users/lmy/asr/ASR_GT_pipline")
sys.path.insert(0, str(BASE_DIR / "Quality"))
sys.path.insert(0, str(BASE_DIR / "Quality"))

from AQ.aq import compute_aq
from TQ.tq import compute_tq
from run_funasr_eval import run_teacher_asr
from utils.text_norm import text_normalize
from jiwer import cer
import jieba


def tokenize_for_wer(text: str) -> str:
    words = jieba.lcut(text)
    return " ".join([w.strip() for w in words if w.strip()])


def decide_final_label(
    text_online: str, text_teacher: str, aq: float, tq_online: float, tq_teacher: float
) -> tuple[str, bool, list[str]]:
    """决策策略（不依赖 GT）"""
    # 计算一致性
    hyp_online = text_normalize(text_online)
    hyp_teacher = text_normalize(text_teacher)
    agreement_cer = cer(hyp_online, hyp_teacher)

    # Rule 1: 音频质量太差,dev里面最差是0.49
    if aq < 0.4:
        return "", False, ["low_aq"]

    # Rule 2: Online 质量高 → 保留
    if tq_online >= 0.6:
        return text_online, True, ["high_online_tq"]

    # Rule 3: Teacher 明显更好 → 替换
    if tq_teacher >= 0.6 and tq_online < 0.6 and agreement_cer > 0.2:
        return text_teacher, True, ["teacher_replace", "teacher_better"]

    # Rule 4: 默认保留 online
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

    # === 2. 运行 Teacher ASR ===
    print(f"🚀 Running Teacher ASR on {args.subset}...")
    teacher_results = run_teacher_asr(data_dir, MODEL_PATH)

    # === 3. 处理每条样本 ===
    manifest = []
    for meta_item in meta_list:
        utt_id = meta_item["utt_id"]
        wav_path = str(data_dir / meta_item["audio_path"])
        text_online = meta_item["text_online"]

        # AQ
        aq_metrics = compute_aq(wav_path)
        aq = aq_metrics["aq"]

        # TQ
        tq_online = compute_tq(text_online)["tq"]
        text_teacher = teacher_results.get(utt_id, {}).get("hyp_fun_asr", "")
        tq_teacher = compute_tq(text_teacher)["tq"]

        # 决策
        text_final, keep, tags = decide_final_label(
            text_online, text_teacher, aq, tq_online, tq_teacher
        )

        # 构建 manifest 记录
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
                if keep and ("high_online_tq" in tags or "default_keep" in tags)
                else float(tq_teacher)
            ),
            "tags": tags,
        }
        manifest.append(record)

    # === 4. 保存结果 ===
    with open(output_dir / "manifest.jsonl", "w", encoding="utf-8") as f:
        for rec in manifest:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"✅ Done! Processed {len(manifest)} samples.")
    print(f"📄 Output saved to {output_dir}/manifest.jsonl")


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
