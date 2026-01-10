# Quality/AQ/run_utmos.py
import os
import json
from pathlib import Path

# === 关键：在导入 utmosv2 之前设置环境变量 ===
# os.environ["HF_HUB_OFFLINE"] = "1"
# os.environ["TIMM_DISABLE_HF"] = "1"  # 👈 禁用 timm 的 Hugging Face 集成
# =========================================
from unittest.mock import patch

EFFNET_LOCAL_BIN = "/calc/users/cisri_shzh_gpu/users/lmy/asr/ASR_GT_pipline/Quality/AQ/timm/tf_efficientnetv2_s.in21k_ft_in1k/pytorch_model.bin"
W2V2_LOCAL_DIR = "/calc/users/cisri_shzh_gpu/users/lmy/asr/models/wav2vec2-base"  # 你的 wav2vec2 路径


def _offline_create_model(model_name, pretrained=False, **kwargs):
    """
    劫持 timm.create_model 和 transformers AutoModel.from_pretrained
    """
    # 1. 处理 EfficientNet-V2
    if model_name == "tf_efficientnetv2_s.in21k_ft_in1k" and pretrained:
        net = timm.create_model(model_name, pretrained=False, **kwargs)
        net.load_state_dict(torch.load(EFFNET_LOCAL_BIN, map_location="cpu"))
        return net

    # 2. 处理 wav2vec2（transformers 侧）
    if model_name == "facebook/wav2vec2-base":
        from transformers import AutoModel, AutoFeatureExtractor

        processor = AutoFeatureExtractor.from_pretrained(
            W2V2_LOCAL_DIR, local_files_only=True
        )
        model = AutoModel.from_pretrained(W2V2_LOCAL_DIR, local_files_only=True)
        # 返回模型对象（UTMOSv2 会自己取 model）
        return model

    # 其余模型保持默认
    return timm.create_model(model_name, pretrained=pretrained, **kwargs)


# 3. 全局打补丁（必须在 import utmosv2 之前）
patch("timm.create_model", side_effect=_offline_create_model).start()
patch(
    "transformers.AutoModel.from_pretrained", side_effect=_offline_create_model
).start()

import utmosv2  # 必须在设置环境变量之后导入！


def extract_utt_id(wav_path: str) -> str:
    return os.path.basename(wav_path).rsplit(".", 1)[0]


def process_directory(wav_dir: Path, output_jsonl: Path, model):
    print(f"🔍 Processing {wav_dir}")
    wav_files = sorted(wav_dir.glob("*.wav"))
    print(f"📁 Found {len(wav_files)} .wav files")

    results = []
    for i, wav_path in enumerate(wav_files):
        utt_id = extract_utt_id(str(wav_path))
        try:
            score = model.predict(input_path=str(wav_path))
            results.append({"utt_id": utt_id, "utmos": round(score, 4)})
        except Exception as e:
            print(f"❌ Failed on {wav_path}: {e}")
            results.append({"utt_id": utt_id, "utmos": None})

        if (i + 1) % 50 == 0:
            print(f"  ✅ Processed {i+1}/{len(wav_files)}")

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"🎉 Done! Saved to {output_jsonl}\n")


def main():
    script_dir = Path(__file__).parent.resolve()
    checkpoint_path = script_dir / "UTMOSv2" / "fold0_s42_best_model.pth"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model not found at {checkpoint_path}")

    # 创建模型
    model = utmosv2.create_model(
        pretrained=True,
        config="fusion_stage3",
        fold=0,
        seed=42,
        checkpoint_path=str(checkpoint_path),
        device="auto",
    )

    base_data = Path("/calc/users/cisri_shzh_gpu/users/lmy/asr/ASR_GT_pipline/data")
    output_dir = script_dir

    process_directory(
        wav_dir=base_data / "dev" / "wavs",
        output_jsonl=output_dir / "dev_utmos.jsonl",
        model=model,
    )
    process_directory(
        wav_dir=base_data / "train_like" / "wavs",
        output_jsonl=output_dir / "train_like_utmos.jsonl",
        model=model,
    )


if __name__ == "__main__":
    main()
