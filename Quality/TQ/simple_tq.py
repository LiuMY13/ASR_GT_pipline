# compute_lm_scores_cer.py
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

# ===== 配置 =====
MODEL_NAME = "uer/gpt2-chinese-cluecorpussmall"
INPUT_FILE = "/calc/users/cisri_shzh_gpu/users/lmy/asr/ASR_GT_pipline/outputs/per_utt_metrics.jsonl"
OUTPUT_FILE = "per_utt_lm_scores_cer.jsonl"

# ===== 加载模型 =====
print(f"Loading LM: {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.eval()

if torch.cuda.is_available():
    model = model.cuda()
    print("✅ Using GPU")
else:
    print("⚠️ Using CPU (slow)")


# ===== 计算平均 token log probability =====
def get_lm_logprob(text: str) -> float:
    if not text or not text.strip():
        return -100.0
    try:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=False,
            add_special_tokens=False,  # GPT-2 不需要 [CLS]/[SEP]
        )
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(input_ids=inputs["input_ids"], labels=inputs["input_ids"])
            loss = outputs.loss.item()
            return -loss  # 越高越好！
    except Exception as e:
        print(f"⚠️ Error on text: '{text[:30]}...' → {e}")
        return -100.0


# ===== 主流程 =====
results = []
with open(INPUT_FILE, "r") as f:
    for line in f:
        item = json.loads(line.strip())
        utt_id = item["utt_id"]

        ref_text = item["ref_cer"]  # 已去标点、小写、无空格
        hyp_text = item["hyp_cer"]

        ref_logprob = get_lm_logprob(ref_text)
        hyp_logprob = get_lm_logprob(hyp_text)

        results.append(
            {
                "utt_id": utt_id,
                "ref_cer": ref_text,
                "hyp_cer": hyp_text,
                "cer": item["cer"],
                "wer": item["wer"],
                "ref_lm_logprob": round(ref_logprob, 4),
                "hyp_lm_logprob": round(hyp_logprob, 4),
                "lm_logprob_gap": round(
                    ref_logprob - hyp_logprob, 4
                ),  # >0 表示 hyp 更差
            }
        )

# ===== 保存结果 =====
with open(OUTPUT_FILE, "w") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"✅ Done! Results saved to {OUTPUT_FILE}")
