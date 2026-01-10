# analyze.py
import json
import pandas as pd
from jiwer import cer
import re
import unicodedata


# ===== 文本标准化（与你的 pipeline 一致）=====
def text_normalize(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[^\u4e00-\u9fa5a-z0-9]", "", text)
    return text


# ===== 加载数据 =====
INPUT_FILE = "/calc/users/cisri_shzh_gpu/users/lmy/asr/ASR_GT_pipline/outputs/dev_full_features.jsonl"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f if line.strip()]

df = pd.DataFrame(data)

# ===== 计算 agreement_cer (online vs teacher) =====
agreement_cers = []
for _, row in df.iterrows():
    hyp_online = text_normalize(row["text_online"])
    hyp_teacher = text_normalize(row["text_teacher"])
    agreement_cers.append(cer(hyp_online, hyp_teacher))
df["agreement_cer"] = agreement_cers

# ===== 1. 基础统计 =====
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
]

print("=== 基础统计 (Descriptive Statistics) ===")
desc_stats = df[features].describe().T
print(desc_stats.round(4))
desc_stats.to_csv("outputs/feature_statistics.csv")

# ===== 2. 相关性分析 =====
print("\n=== 与 online_cer 的 Pearson 相关系数 ===")
correlation = df[features].corr()["online_cer"].sort_values(key=abs, ascending=False)
print(correlation.round(4))
correlation.to_csv("outputs/correlation_with_online_cer.csv")


# ===== 3. 决策策略 =====
def decide_keep(record: dict) -> tuple[bool, list[str], str]:
    # 计算 agreement_cer
    hyp_online = text_normalize(record["text_online"])
    hyp_teacher = text_normalize(record["text_teacher"])
    agreement_cer = cer(hyp_online, hyp_teacher)

    # Rule 1: 音频硬过滤（极端情况）
    if record["aq"] < 0.5:
        return False, ["low_aq"], ""

    # Rule 2: Online 质量高 → 保留（TQ > 0.75 或 CER 估计低）
    if record["hyp_online_tq"] >= 0.75:
        return True, ["high_online_tq"], record["text_online"]

    # Rule 3: Teacher 明显更好 → 替换
    # 条件：Teacher TQ 高 + agreement_cer 大（说明 Online 错了）
    if (
        record["hyp_teacher_tq"] >= 0.75
        and record["hyp_online_tq"] < 0.6
        and agreement_cer > 0.2
    ):
        return True, ["teacher_replace", "teacher_better"], record["text_teacher"]

    # Rule 4: 默认保留 online（即使中等质量）
    return True, ["default_keep"], record["text_online"]


# 应用策略
results = []
for _, row in df.iterrows():
    keep, tags, text_final = decide_keep(row.to_dict())

    # 计算 final_cer (vs GT)
    if keep:
        ref_norm = text_normalize(row["text_gt"])
        hyp_norm = text_normalize(text_final)
        final_cer = cer(ref_norm, hyp_norm)
    else:
        final_cer = 1.0

    results.append(
        {
            "utt_id": row["utt_id"],
            "keep": keep,
            "tags": tags,
            "text_final": text_final,
            "final_cer": final_cer,
            "duration_sec": row["duration_sec"],
        }
    )

# ===== 4. 策略效果评估 =====
result_df = pd.DataFrame(results)

total_duration = result_df["duration_sec"].sum()
weighted_cer = (
    result_df["final_cer"] * result_df["duration_sec"]
).sum() / total_duration
coverage = result_df["keep"].mean()
baseline_weighted_cer = (df["online_cer"] * df["duration_sec"]).sum() / df[
    "duration_sec"
].sum()

print(f"\n=== 策略效果 ===")
print(f"Baseline Weighted CER: {baseline_weighted_cer:.4f}")
print(f"Final Weighted CER:    {weighted_cer:.4f}")
print(
    f"CER Reduction:         {(baseline_weighted_cer - weighted_cer):.4f} ({(baseline_weighted_cer - weighted_cer)/baseline_weighted_cer*100:.1f}%)"
)
print(
    f"Coverage:              {coverage:.2%} ({result_df['keep'].sum()}/{len(result_df)})"
)

# ===== 5. 保存 manifest.jsonl =====
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
                if "high_online_tq" in row["tags"] or "medium_quality" in row["tags"]
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
print(f"- outputs/manifest.jsonl")
