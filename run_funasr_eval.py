# from pathlib import Path
# from jiwer import cer, wer
# from funasr import AutoModel
# import torch
# import os
# import json

# # Text normalize
# try:
#     from utils.text_norm import text_normalize
# except ImportError:
#     import unicodedata, re

#     def text_normalize(text: str) -> str:
#         if not isinstance(text, str):
#             return ""
#         text = unicodedata.normalize("NFKC", text).lower()
#         text = re.sub(r"[^\u4e00-\u9fa5a-z0-9]", " ", text)
#         return re.sub(r"\s+", " ", text).strip()


# import jieba


# def tokenize_for_wer(text: str) -> str:
#     words = jieba.lcut(text)
#     return " ".join([w.strip() for w in words if w.strip()])


# def compute_per_utt_cer_wer(ref: str, hyp: str) -> tuple[float, float]:
#     try:
#         c = cer(ref, hyp)
#     except:
#         c = 1.0
#     try:
#         w = wer(ref, hyp)
#     except:
#         w = 1.0
#     return c, w


# def run_teacher_asr(dev_dir: Path, model_path: str) -> dict:
#     """返回 {utt_id: {hyp_fun_asr, cer, wer, ...}}"""
#     # Load GT
#     gt_map = {}
#     with open(dev_dir / "gt.jsonl", "r", encoding="utf-8") as f:
#         for line in f:
#             if line.strip():
#                 item = json.loads(line)
#                 gt_map[item["utt_id"]] = item["text_gt"]

#     # Load model
#     model = AutoModel(
#         model=model_path,
#         trust_remote_code=True,
#         device="cuda:0" if torch.cuda.is_available() else "cpu",
#         disable_update=True,
#         remote_code="./Fun-ASR/model.py",
#     )

#     results = {}
#     with open(dev_dir / "meta.jsonl", "r", encoding="utf-8") as f_meta:
#         for line in f_meta:
#             if not line.strip():
#                 continue
#             meta_item = json.loads(line)
#             utt_id = meta_item["utt_id"]
#             if utt_id not in gt_map:
#                 continue

#             wav_path = dev_dir / meta_item["audio_path"]
#             if not wav_path.exists():
#                 continue

#             ref_raw = gt_map[utt_id]

#             # ASR
#             try:
#                 res = model.generate(
#                     input=str(wav_path), batch_size=1, language="zh", itn=True
#                 )
#                 hyp_raw = res[0]["text"] if res else ""
#             except:
#                 hyp_raw = ""

#             # Normalize
#             ref_cer = text_normalize(ref_raw)
#             hyp_cer = text_normalize(hyp_raw)
#             if not ref_cer:
#                 continue

#             ref_wer = tokenize_for_wer(ref_cer)
#             hyp_wer = tokenize_for_wer(hyp_cer)

#             utt_cer, _ = compute_per_utt_cer_wer(ref_cer, hyp_cer)
#             _, utt_wer = compute_per_utt_cer_wer(ref_wer, hyp_wer)

#             results[utt_id] = {
#                 "hyp_fun_asr": hyp_raw,
#                 "cer": round(utt_cer, 4),
#                 "wer": round(utt_wer, 4),
#             }

#     return results
# run_funasr_eval.py
import json
from pathlib import Path
from jiwer import cer, wer
from funasr import AutoModel
import torch
import os

# 文本 normalize（与 eval_baseline 一致）
try:
    from utils.text_norm import text_normalize
except ImportError:
    import unicodedata, re

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


def compute_per_utt_cer_wer(ref: str, hyp: str) -> tuple[float, float]:
    try:
        c = cer(ref, hyp)
    except:
        c = 1.0
    try:
        w = wer(ref, hyp)
    except:
        w = 1.0
    return c, w


def run_teacher_asr(dev_dir: Path, model_path: str, has_gt: bool = True) -> dict:
    """
    对 dev_dir 运行 Teacher ASR
    Args:
        dev_dir: 数据目录（含 meta.jsonl 和可选 gt.jsonl）
        model_path: 模型路径
        has_gt: 是否有 GT（仅 dev 有，train_like/test 没有）
    Returns:
        {utt_id: {"hyp_fun_asr": str, "cer": float, "wer": float}}
        （无 GT 时 cer/wer 为 None）
    """
    # 只在有 GT 时加载
    gt_map = {}
    if has_gt:
        gt_file = dev_dir / "gt.jsonl"
        if gt_file.exists():
            with open(gt_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        gt_map[item["utt_id"]] = item["text_gt"]
        else:
            print(f"⚠️ Warning: {gt_file} not found. Proceeding without GT.")
            has_gt = False

    # 加载模型
    model = AutoModel(
        model=model_path,
        trust_remote_code=True,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        disable_update=True,
        remote_code="./Fun-ASR/model.py",
    )

    results = {}
    with open(dev_dir / "meta.jsonl", "r", encoding="utf-8") as f_meta:
        for line in f_meta:
            if not line.strip():
                continue
            meta_item = json.loads(line)
            utt_id = meta_item["utt_id"]

            wav_path = dev_dir / meta_item["audio_path"]
            if not wav_path.exists():
                print(f"⚠️ Audio not found: {wav_path}")
                continue

            # ASR 推理
            try:
                res = model.generate(
                    input=str(wav_path), batch_size=1, language="zh", itn=True
                )
                hyp_raw = res[0]["text"] if res else ""
            except Exception as e:
                print(f"❌ ASR failed on {utt_id}: {e}")
                hyp_raw = ""

            # 初始化结果
            result = {"hyp_fun_asr": hyp_raw}

            # 仅在有 GT 时计算 CER/WER
            if has_gt and utt_id in gt_map:
                ref_raw = gt_map[utt_id]
                ref_cer = text_normalize(ref_raw)
                hyp_cer = text_normalize(hyp_raw)
                if ref_cer:
                    ref_wer = tokenize_for_wer(ref_cer)
                    hyp_wer = tokenize_for_wer(hyp_cer)
                    utt_cer, _ = compute_per_utt_cer_wer(ref_cer, hyp_cer)
                    _, utt_wer = compute_per_utt_cer_wer(ref_wer, hyp_wer)
                    result["cer"] = round(utt_cer, 4)
                    result["wer"] = round(utt_wer, 4)
                else:
                    result["cer"] = None
                    result["wer"] = None
            else:
                result["cer"] = None
                result["wer"] = None

            results[utt_id] = result

    return results
