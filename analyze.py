# analyze.py (with WER support)
import json
import pandas as pd
from jiwer import cer, wer
import re
import unicodedata
import jieba
from utils.text_norm import text_normalize

# # ===== 文本标准化 =====
# def text_normalize(text: str) -> str:
#     if not isinstance(text, str):
#         return ""
#     text = unicodedata.normalize("NFKC", text).lower()
#     text = re.sub(r"[^\u4e00-\u9fa5a-z0-9]", "", text)
#     return text


# ===== WER 分词 =====
def tokenize_for_wer(text: str) -> str:
    words = jieba.lcut(text)
    return " ".join([w.strip() for w in words if w.strip()])


# ===== 加载数据 =====
INPUT_FILE = "/calc/users/cisri_shzh_gpu/users/lmy/asr/ASR_GT_pipline/outputs/dev_full_features.jsonl"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f if line.strip()]

df = pd.DataFrame(data)

# ===== 计算 agreement_cer / agreement_wer =====
agreement_cers = []
agreement_wers = []
for _, row in df.iterrows():
    hyp_online = text_normalize(row["text_online"])
    hyp_teacher = text_normalize(row["text_teacher"])
    agreement_cers.append(cer(hyp_online, hyp_teacher))

    # WER 需要分词
    hyp_online_wer = tokenize_for_wer(hyp_online)
    hyp_teacher_wer = tokenize_for_wer(hyp_teacher)
    agreement_wers.append(wer(hyp_online_wer, hyp_teacher_wer))

df["agreement_cer"] = agreement_cers
df["agreement_wer"] = agreement_wers

# ===== 1. 基础统计（新增 WER 相关字段）=====
features = [
    "online_cer",
    "online_wer",
    "teacher_cer",
    "teacher_wer",
    "aq",
    "snr_db",
    "speech_ratio",
    "clip_ratio",
    "hyp_online_tq",
    "hyp_teacher_tq",
    "ref_tq",
    "hyp_online_lm_logprob",
    "hyp_teacher_lm_logprob",
    "ref_lm_logprob",
    "agreement_cer",
    "agreement_wer",  # ← 新增
]

print("=== 基础统计 ===")
desc_stats = df[features].describe().T
print(desc_stats.round(4))
desc_stats.to_csv("outputs/feature_statistics.csv")

# ===== 2. 相关性分析（同时看 CER 和 WER）=====
print("\n=== 与 online_cer 的相关性 ===")
corr_cer = df[features].corr()["online_cer"].sort_values(key=abs, ascending=False)
print(corr_cer.round(4))

print("\n=== 与 online_wer 的相关性 ===")
corr_wer = df[features].corr()["online_wer"].sort_values(key=abs, ascending=False)
print(corr_wer.round(4))

# 保存相关性
corr_cer.to_csv("outputs/correlation_with_online_cer.csv")
corr_wer.to_csv("outputs/correlation_with_online_wer.csv")


# ===== 3. 决策策略（不变）=====
def decide_keep(record: dict) -> tuple[bool, list[str], str]:
    hyp_online = text_normalize(record["text_online"])
    hyp_teacher = text_normalize(record["text_teacher"])
    agreement_cer = cer(hyp_online, hyp_teacher)

    if record["aq"] < 0.5:
        return False, ["low_aq"], ""
    if record["hyp_online_tq"] >= 0.7:
        return True, ["high_online_tq"], record["text_online"]
    if (
        record["hyp_teacher_tq"] >= 0.6
        and record["hyp_online_tq"] < 0.6
        and agreement_cer > 0.2
    ):
        return True, ["teacher_replace", "teacher_better"], record["text_teacher"]
    return True, ["default_keep"], record["text_online"]


# ===== 4. 应用策略并计算 CER/WER =====
results = []
for _, row in df.iterrows():
    keep, tags, text_final = decide_keep(row.to_dict())

    if keep:
        # CER
        ref_cer = text_normalize(row["text_gt"])
        hyp_cer = text_normalize(text_final)
        final_cer = cer(ref_cer, hyp_cer)

        # WER
        ref_wer = tokenize_for_wer(ref_cer)
        hyp_wer = tokenize_for_wer(hyp_cer)
        final_wer = wer(ref_wer, hyp_wer)
    else:
        final_cer = 1.0
        final_wer = 1.0

    results.append(
        {
            "utt_id": row["utt_id"],
            "keep": keep,
            "tags": tags,
            "text_final": text_final,
            "final_cer": final_cer,
            "final_wer": final_wer,
            "duration_sec": row["duration_sec"],
        }
    )

# ===== 5. 策略效果评估（CER + WER）=====
result_df = pd.DataFrame(results)

# Weighted CER
baseline_cer = (df["online_cer"] * df["duration_sec"]).sum() / df["duration_sec"].sum()
final_cer = (result_df["final_cer"] * result_df["duration_sec"]).sum() / result_df[
    "duration_sec"
].sum()

# Weighted WER
baseline_wer = (df["online_wer"] * df["duration_sec"]).sum() / df["duration_sec"].sum()
final_wer = (result_df["final_wer"] * result_df["duration_sec"]).sum() / result_df[
    "duration_sec"
].sum()

coverage = result_df["keep"].mean()

print(f"\n=== 策略效果 ===")
print(
    f"Baseline Weighted CER: {baseline_cer:.4f} → Final: {final_cer:.4f} (Δ: {baseline_cer-final_cer:+.4f})"
)
print(
    f"Baseline Weighted WER: {baseline_wer:.4f} → Final: {final_wer:.4f} (Δ: {baseline_wer-final_wer:+.4f})"
)
print(f"Coverage: {coverage:.2%} ({result_df['keep'].sum()}/{len(result_df)})")

# ===== 6. 保存 manifest.jsonl =====
manifest = []
for _, row in result_df.iterrows():
    orig = df[df["utt_id"] == row["utt_id"]].iloc[0]
    manifest.append(
        {
            "utt_id": row["utt_id"],
            "audio_path": orig["audio_path"],
            "text_online": orig["text_online"],
            "text_teacher": orig["text_teacher"],
            "text_final": row["text_final"],
            "keep": row["keep"],
            "aq": float(orig["aq"]),
            "tq": (
                float(orig["hyp_online_tq"])
                if "high_online_tq" in row["tags"] or "default_keep" in row["tags"]
                else float(orig["hyp_teacher_tq"])
            ),
            "tags": row["tags"],
        }
    )

with open("outputs/manifest.jsonl", "w", encoding="utf-8") as f:
    for rec in manifest:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"\n✅ 输出文件:")
print(f"- outputs/feature_statistics.csv")
print(f"- outputs/correlation_with_online_cer.csv")
print(f"- outputs/correlation_with_online_wer.csv")  # ← 新增
print(f"- outputs/manifest.jsonl")
