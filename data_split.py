import os
import json
import shutil
import random
from pathlib import Path


def main():
    org_dev_dir = Path("interview_data/org_dev")
    output_root = Path("data")  # 输出到 data/train_like 和 data/dev

    shutil.rmtree(output_root, ignore_errors=True)
    (output_root / "train_like" / "wavs").mkdir(parents=True)
    (output_root / "dev" / "wavs").mkdir(parents=True)

    # 1. 读取 meta.jsonl
    meta_list = []
    with open(org_dev_dir / "meta.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                meta_list.append(json.loads(line))

    # 2. 读取 gt.jsonl 到 dict
    gt_dict = {}
    with open(org_dev_dir / "gt.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                gt_dict[item["utt_id"]] = item["text_gt"]

    # 3. 按 utt_id 对齐（确保都有 GT）
    aligned_data = []
    for meta in meta_list:
        utt_id = meta["utt_id"]
        if utt_id in gt_dict:
            meta["text_gt"] = gt_dict[utt_id]
            aligned_data.append(meta)
        else:
            print(f"Warning: {utt_id} has no GT, skipped.")

    # 4. 随机打乱
    random.seed(42)  # 可复现
    random.shuffle(aligned_data)

    # 5. 划分 2:1 → train_like : dev
    n_total = len(aligned_data)
    n_train = int(n_total * 2 / 3)
    train_data = aligned_data[:n_train]
    dev_data = aligned_data[n_train:]

    print(f"Total samples: {n_total}")
    print(f"Train-like: {len(train_data)}")
    print(f"Dev: {len(dev_data)}")

    # 6. 写 train_like (只有 meta，无 GT)
    train_meta_path = output_root / "train_like" / "meta.jsonl"
    with open(train_meta_path, "w", encoding="utf-8") as f:
        for item in train_data:
            # 移除 text_gt
            item_clean = {k: v for k, v in item.items() if k != "text_gt"}
            f.write(json.dumps(item_clean, ensure_ascii=False) + "\n")
            # 复制音频
            src_wav = org_dev_dir / item["audio_path"]
            dst_wav = output_root / "train_like" / item["audio_path"]
            shutil.copy2(src_wav, dst_wav)

    # 7. 写 dev (meta + gt)
    dev_meta_path = output_root / "dev" / "meta.jsonl"
    dev_gt_path = output_root / "dev" / "gt.jsonl"
    with open(dev_meta_path, "w", encoding="utf-8") as f_meta, open(
        dev_gt_path, "w", encoding="utf-8"
    ) as f_gt:
        for item in dev_data:
            # meta.jsonl (不含 text_gt)
            item_meta = {k: v for k, v in item.items() if k != "text_gt"}
            f_meta.write(json.dumps(item_meta, ensure_ascii=False) + "\n")
            # gt.jsonl
            f_gt.write(
                json.dumps(
                    {"utt_id": item["utt_id"], "text_gt": item["text_gt"]},
                    ensure_ascii=False,
                )
                + "\n"
            )
            # 复制音频
            src_wav = org_dev_dir / item["audio_path"]
            dst_wav = output_root / "dev" / item["audio_path"]
            shutil.copy2(src_wav, dst_wav)

    print("✅ Split completed! Output in 'data/' directory.")


if __name__ == "__main__":
    main()
