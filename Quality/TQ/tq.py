# Quality/TQ/simple_tq.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 全局缓存
_TQ_MODEL = None
_TQ_TOKENIZER = None


def compute_tq(text: str, model_path: str = "uer/gpt2-chinese-cluecorpussmall") -> dict:
    global _TQ_MODEL, _TQ_TOKENIZER

    if not text or not text.strip():
        return {"tq": 0.0, "lm_logprob": -100.0}

    if _TQ_MODEL is None:
        print(f"Loading LM: {model_path} ...")
        _TQ_TOKENIZER = AutoTokenizer.from_pretrained(model_path)
        _TQ_MODEL = AutoModelForCausalLM.from_pretrained(model_path)
        _TQ_MODEL.eval()
        if torch.cuda.is_available():
            _TQ_MODEL = _TQ_MODEL.cuda()
        print("✅ TQ model loaded.")

    try:
        inputs = _TQ_TOKENIZER(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=False,
            add_special_tokens=False,
        )
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = _TQ_MODEL(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss.item()
            lm_logprob = -loss

        tq = max(0.0, min(1.0, (lm_logprob + 8.0) / 6.0))
        return {"tq": round(tq, 4), "lm_logprob": round(lm_logprob, 4)}
    except Exception as e:
        print(f"⚠️ TQ failed: {e}")
        return {"tq": 0.0, "lm_logprob": -100.0}
