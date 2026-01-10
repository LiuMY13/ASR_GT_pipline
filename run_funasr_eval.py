# run_funasr_eval.py
import json
from pathlib import Path
from jiwer import cer, wer
from funasr import AutoModel
import torch
import os

# 假设 utils.text_norm 存在；若不存在，请确保 text_normalize 函数定义在此
try:
    from utils.text_norm import text_normalize
except ImportError:
    # 如果没有 utils/text_norm.py，使用简单 normalize
    import unicodedata
    import re

    def text_normalize(text: str) -> str:
        """Simple text normalization: NFKC + lower + remove punctuation + collapse whitespace"""
        if not isinstance(text, str):
            return ""
        text = unicodedata.normalize("NFKC", text)
        text = text.lower()
        # Remove all punctuation and non-alphanumeric (keep Chinese, English, digits)
        text = re.sub(r"[^\u4e00-\u9fa5a-z0-9]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text


import jieba


def tokenize_for_wer(text: str) -> str:
    words = jieba.lcut(text)
    words = [word.strip() for word in words if word.strip()]
    return " ".join(words)


def compute_per_utt_cer_wer(ref: str, hyp: str) -> tuple[float, float]:
    try:
        c = cer(ref, hyp)
    except Exception:
        c = 1.0
    try:
        w = wer(ref, hyp)
    except Exception:
        w = 1.0
    return c, w


def main():
    # === 路径配置 ===
    BASE_DIR = Path("/calc/users/cisri_shzh_gpu/users/lmy/asr/ASR_GT_pipline")
    data_root = BASE_DIR / "data"
    output_dir = BASE_DIR / "outputs"
    output_dir.mkdir(exist_ok=True)

    dev_dir = data_root / "dev"
    MODEL_LOCAL_PATH = str(BASE_DIR / "FunAudioLLM/Fun-ASR-Nano-2512")  # 本地模型路径

    # === 加载 GT ===
    gt_map = {}
    with open(dev_dir / "gt.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                gt_map[item["utt_id"]] = item["text_gt"]

    # === 加载 FunASR 模型（本地路径）===
    print(f"Loading FunASR model from: {MODEL_LOCAL_PATH}")
    if not os.path.exists(MODEL_LOCAL_PATH):
        raise FileNotFoundError(f"Model path not found: {MODEL_LOCAL_PATH}")

    model = AutoModel(
        model=MODEL_LOCAL_PATH,
        trust_remote_code=True,  # 必须为 True 以加载 model.py
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        disable_update=True,  # 关闭版本检查
        remote_code="/calc/users/cisri_shzh_gpu/users/lmy/asr/ASR_GT_pipline/Fun-ASR/model.py",
    )
    print("✅ FunASR model loaded successfully.")

    # === 初始化结果容器 ===
    per_utt_list = []
    refs_cer_all = []
    hyps_cer_all = []
    refs_wer_all = []
    hyps_wer_all = []

    # === 遍历 dev/meta.jsonl ===
    with open(dev_dir / "meta.jsonl", "r", encoding="utf-8") as f_meta:
        for line in f_meta:
            if not line.strip():
                continue
            meta_item = json.loads(line)
            utt_id = meta_item["utt_id"]
            if utt_id not in gt_map:
                continue

            wav_path = dev_dir / meta_item["audio_path"]
            if not wav_path.exists():
                print(f"⚠️ Audio not found: {wav_path}")
                continue

            ref_raw = gt_map[utt_id]

            # === ASR 推理 ===
            try:
                res = model.generate(
                    input=str(wav_path),
                    batch_size=1,
                    language="zh",  # 注意：本地模型通常用 "zh" 而非 "中文"
                    itn=True,  # 启用逆文本归一化
                )
                hyp_raw = res[0]["text"] if res else ""
            except Exception as e:
                print(f"❌ ASR failed on {utt_id}: {e}")
                hyp_raw = ""

            # === 文本标准化 ===
            ref_cer = text_normalize(ref_raw)
            hyp_cer = text_normalize(hyp_raw)
            if not ref_cer:
                continue

            ref_wer = tokenize_for_wer(ref_cer)
            hyp_wer = tokenize_for_wer(hyp_cer)

            # === 计算指标 ===
            utt_cer, _ = compute_per_utt_cer_wer(ref_cer, hyp_cer)
            _, utt_wer = compute_per_utt_cer_wer(ref_wer, hyp_wer)

            # === 保存结果 ===
            per_utt_list.append(
                {
                    "utt_id": utt_id,
                    "ref_raw": ref_raw,
                    "hyp_fun_asr": hyp_raw,
                    "ref_cer": ref_cer,
                    "hyp_cer": hyp_cer,
                    "ref_wer": ref_wer,
                    "hyp_wer": hyp_wer,
                    "cer": round(utt_cer, 4),
                    "wer": round(utt_wer, 4),
                }
            )

            refs_cer_all.append(ref_cer)
            hyps_cer_all.append(hyp_cer)
            refs_wer_all.append(ref_wer)
            hyps_wer_all.append(hyp_wer)

    # === 计算整体指标 ===
    overall_cer = cer(refs_cer_all, hyps_cer_all)
    overall_wer = wer(refs_wer_all, hyps_wer_all)

    # === 保存 per-utterance 结果 ===
    with open(output_dir / "dev_fun_asr.jsonl", "w", encoding="utf-8") as f:
        for record in per_utt_list:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # === 保存整体分数 ===
    score_result = {
        "fun_asr_cer": round(overall_cer, 4),
        "fun_asr_wer": round(overall_wer, 4),
        "num_samples": len(per_utt_list),
        "model_path": MODEL_LOCAL_PATH,
        "language": "zh",
        "itn": True,
        "normalize_rules": "NFKC + lower + remove punctuation + collapse whitespace",
        "wer_tokenization": "jieba for Chinese",
    }

    with open(output_dir / "dev_score_fun_asr.json", "w", encoding="utf-8") as f:
        json.dump(score_result, f, ensure_ascii=False, indent=2)

    print(f"✅ Overall CER: {overall_cer:.4f}")
    print(f"✅ Overall WER: {overall_wer:.4f}")
    print(f"📊 Processed {len(per_utt_list)} samples")
    print(f"📄 Results saved to {output_dir}")


if __name__ == "__main__":
    main()
