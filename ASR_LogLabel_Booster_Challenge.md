# 语音算法实习生挑战：LogLabel Booster（线上日志自动清洗与伪标注）

**建议时长：** 8–10 小时  
**提交形式：** 私有 GitHub 仓库（邀请面试官为 Collaborator）  
**目标关键词：** 数据清洗 / 伪标注 / 质量打分 / 自动化 Pipeline / 可复现 / 可解释

---

## 1. 项目背景

真实业务里，ASR 的迭代往往不是“换个模型就好了”，而是要持续从线上回流中挖出**更干净、更有价值、标签更可信**的数据。

你现在拿到的是一批“线上日志”样本：

- `audio`：用户真实语音（已切分为 utterance 或给出切分信息）
- `text_online`：线上 ASR 模型的转录结果（**噪声标签**，会有错字漏字、乱断句、口语、混语、噪声段误识别等问题）
- `metadata`：一些基础信息（时长、场景、设备等，可能不完整）

你的任务是搭一套最小可行的自动化 pipeline：**清洗 + 重打标签 + 输出可训练数据**，并用离线指标证明“标签确实更好了”。

---

## 2. 任务目标

请实现一个命令行工具：`loglabel_booster`

### 核心输入

给定一个数据目录 `data/`，其中包含三份子集：

- `train_like/`：模拟线上回流（无 GT，仅有 `text_online`）
- `dev/`：带少量 `text_gt`（用于你调参 & 自测，不可在 pipeline 里“偷看”做规则）
- `test/`：不提供 `text_gt`（我们评测用）

> 我们会提供数据包（结构见下），你不需要自己下载公开数据。

### 核心输出

你需要产出一个“可用于训练/迭代”的清洗后 manifest，并给出每条样本的最终标签与质量信息：

1) `outputs/manifest.jsonl`（每行一个 utterance 的结构化结果）  
2) `outputs/report.md`（你的方法说明 + 实验结果 + 关键统计图/表）  
3) `outputs/dev_score.json`（跑在 dev 上的量化指标，便于我们快速看效果）

---

## 3. 数据格式（我们提供）

目录结构示例：

```
data/
  train_like/
    wavs/
      utt_000001.wav
      ...
    meta.jsonl
  dev/
    wavs/
    meta.jsonl          # 含 text_online
    gt.jsonl            # 含 text_gt（仅 dev 提供）
  test/
    wavs/
    meta.jsonl          # 仅含 text_online
```

`meta.jsonl` 每行示例（字段可能略有增减）：

```json
{
  "utt_id": "utt_000001",
  "audio_path": "wavs/utt_000001.wav",
  "duration_sec": 5.32,
  "text_online": "jin tian tian qi hen hao",
  "scene": "farfield",
  "device": "mobile"
}
```

`gt.jsonl`（仅 dev）示例：

```json
{
  "utt_id": "utt_000001",
  "text_gt": "今天天气很好"
}
```

---

## 4. 你需要完成的核心功能

### 4.1 必做：数据清洗与质量打分（Quality Scoring）

请至少实现 **2 类以上**的质量信号，并输出到 manifest：

- **音频侧（至少 1 类）**：如时长过滤、静音占比、能量、clipping 检测、SNR proxy、简单音频分类（语音/音乐/噪声）等  
- **文本/模型侧（至少 1 类）**：如 teacher 置信度、N-best 一致性、online vs teacher 差异度、LM plausibility（困惑度/打分）等

输出建议字段（你可自行扩展）：

- `aq`：音频质量分（0–1）
- `tq`：转录质量分（0–1）
- `keep`：是否保留做训练（bool）
- `drop_reason`：丢弃原因（可多标签，如 `["too_short","music_prob_high"]`）

---

### 4.2 必做：使用开源 ASR 模型做伪标注（Relabel）

你需要至少接入 **1 个开源 ASR 模型**，对音频重新转写，得到 `text_teacher`，并基于策略生成最终标签 `text_final`。

你可以自由选择模型（不限于）：

- Whisper（推荐用 `faster-whisper` 便于速度）
- Wav2Vec2 / HuBERT CTC
- FunASR / WeNet / Paraformer 等开源 ASR

**关键要求：**

- 你不能简单“全量用 teacher 覆盖 online”，需要给出“何时覆盖、何时保留、何时丢弃”的策略，并在 report 里解释。
- 建议引入“agreement/对齐/置信”等硬闸门，避免把 teacher 的错当真。

`manifest.jsonl` 每行至少包含：

```json
{
  "utt_id": "...",
  "audio_path": "...",
  "text_online": "...",
  "text_teacher": "...",
  "text_final": "...",
  "keep": true,
  "aq": 0.83,
  "tq": 0.71,
  "tags": ["teacher_replace", "high_agreement"]
}
```

---

### 4.3 必做：文本规范化（Text Normalization）

为了公平评测，你需要实现一个可复现的 `text_normalize()`（并在 report 说明规则），至少覆盖：

- 全半角/空白清理
- 标点策略（保留/去除/统一）
- 数字表达（可选，但加分）
- 英文大小写（若有）

**评测会在同一套 normalize 口径下进行**，所以你要保证代码里有明确版本/规则说明。

---

### 4.4 必做：在 dev 上给出量化改进

你需要在 `dev` 集上对比：

- baseline：`text_online` 的 CER/WER
- your output：`text_final` 的 CER/WER

并把结果写入：

- `outputs/dev_score.json`
- `outputs/report.md`

---

## 5. 评分标准（我们会在隐藏 test 上跑）

我们主要关注：**标签更准，同时不要靠“全丢弃”刷分**。

### 5.1 主指标：加权错误率（考虑覆盖）

对每条 test utterance：

- 若 `keep=false` 或 `text_final` 为空：该条错误率记为 `1.0`
- 否则错误率为 `CER(text_final, text_gt)`（或 WER，按数据语言为主）

整体错误率按 `duration_sec` 加权平均：

```
Error = sum(duration_i * err_i) / sum(duration_i)
Score = 100 * (1 - Error)
```

我们会同时计算 baseline（用 `text_online`），看你的 `Score` 提升幅度。

---

### 5.2 工程与可复现（重要）

- 一条命令可跑通（从原始 `data/` 到 `outputs/`）
- 日志清晰、结构化输出
- 规则/阈值可配置（yaml/json/config.py 都行）
- 不允许对某些 `utt_id` 做硬编码特判（No Hard-coding）

---

### 5.3 可解释性与业务思维

`report.md` 里要回答：

- 你认为线上日志里最主要的脏数据类型是什么？你怎么识别？
- 你是怎么决定“保留/丢弃/覆盖 online/融合”的？
- 你做了哪些 ablation（哪怕非常粗糙）？

---

## 6. 提交指南（Submission Guidelines）

1) 创建 **Private** GitHub repo  
2) 邀请面试官 GitHub 账号为 Collaborator  
3) **原子化提交**：每完成一个模块就 commit（我们会看时间线）  
4) 根目录需要 `README.md`，包含：  
   - 环境安装方式（conda/pip）
   - 如何运行（示例命令）
   - 你用到的开源模型与版本（以及是否需要 GPU）
   - 你输出的文件说明  
5) 需要包含一个最小可运行 demo，例如：

```bash
python -m loglabel_booster.run \
  --input_dir data/ \
  --output_dir outputs/ \
  --subset dev \
  --config configs/default.yaml
```

---

## 7. 加分项（Bonus Challenges）

### A. 更“先进”的标签可信度闸门

任选其一即可（做出来很加分）：

- **forced alignment**（CTC 对齐/MFA/CTC-segmentation 任意）：对齐失败则降权或丢弃  
- **多 teacher 一致性**：两种模型转写，做共识（ROVER/投票/最短编辑距离中心）  
- **N-best / dropout 一致性**：同一模型多次解码一致性作为置信

### B. 数据分桶 + 样本加权

输出 `bucket` 和 `weight`，例如：

- `gold`：直接用（w=1.0）
- `relabel`：teacher 替换（w=0.7）
- `hard`：疑难但高价值（w=0.5）
- `drop`：丢弃（w=0.0）

### C. 速度与可扩展

- 多进程/批处理加速
- 支持断点续跑（cache 中间结果）
- 资源受限模式（CPU-only 也能跑通）

### D. 进一步验证“对训练有用”（可选）

如果你有 GPU 和时间：

- 用你产出的 `train_like` 清洗结果做一个小规模 fine-tune（哪怕只跑 1–2 epoch）
- 在 dev 上对比微调前后 WER/CER（能跑通闭环非常加分）

---

## 8. 时间管理建议（Scope & Time）

你不需要做一个“生产级”系统，我们更看重 **MVP 的闭环**：

**建议时间分配：**

1. 0.5h：跑通读取数据 + normalize + baseline 评测  
2. 2–3h：接入 teacher ASR + 导出 `text_teacher`  
3. 2–3h：设计 quality signals + 决策策略（keep/drop/replace）  
4. 1h：产出 report（分桶统计、典型 case）  
5. 剩余时间：bonus（对齐/多 teacher/加速/加权）
