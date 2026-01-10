# Teacher_ASR/run_teacher.py
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))  # 添加项目根目录到 path

import json
from pathlib import Path
from Teacher_ASR.asr_model import FasterWhisperASR
from utils.text_norm import text_normalize


def process_subset(input_dir: Path, output_dir: Path, subset: str, asr_model):
    meta_path = input_dir / subset / "meta.jsonl"
    wavs_dir = input_dir / subset / "wavs"
    output_path = output_dir / f"{subset}_with_teacher.jsonl"

    with open(meta_path, "r", encoding="utf-8") as fin, open(
        output_path, "w", encoding="utf-8"
    ) as fout:

        for line in fin:
            if not line.strip():
                continue
            item = json.loads(line)
            utt_id = item["utt_id"]
            audio_file = wavs_dir / item["audio_path"]

            # 获取 online 文本
            text_online = item["text_online"]

            # ASR 推理
            try:
                text_teacher_raw = asr_model.transcribe(str(audio_file))
                text_teacher = text_normalize(text_teacher_raw)
            except Exception as e:
                print(f"❌ ASR failed for {utt_id}: {e}")
                text_teacher = ""

            # 写出结果
            out_item = {
                "utt_id": utt_id,
                "audio_path": item["audio_path"],
                "duration_sec": item.get("duration_sec", 0.0),
                "text_online": text_online,
                "text_teacher": text_teacher,
                "scene": item.get("scene", ""),
                "device": item.get("device", ""),
            }
            fout.write(json.dumps(out_item, ensure_ascii=False) + "\n")

    print(f"✅ Finished {subset}, saved to {output_path}")


def main():
    input_dir = Path("../data")
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    # 初始化 ASR（使用 medium 模型）
    asr = FasterWhisperASR(model_size="medium", device="auto")

    # 处理 dev 和 train_like
    for subset in ["dev", "train_like"]:
        process_subset(input_dir, output_dir, subset, asr)


if __name__ == "__main__":
    main()
