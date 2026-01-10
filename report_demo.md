# Report demo
# Data Split
从dev里面随机划分，train
train:dev = 2:1 seed:42

# text normalization
为了公平评测，你需要实现一个可复现的 `text_normalize()`（并在 report 说明规则），至少覆盖：

- 全半角/空白清理
- 标点策略（保留/去除/统一）
- 数字表达（可选，但加分）
- 英文大小写（若有）


# eval baseline
wer和cer输入有区别


# Teacher ASR:
使用的是Fast-Whisper，效果变差。一会儿再换teacher ASR
📊 Teacher ASR Evaluation:
  Baseline CER: 0.0758 → Teacher CER: 0.1165 (Δ=-0.0407)
  Baseline WER: 0.1507 → Teacher WER: 0.2039 (Δ=-0.0532)
效果太差，不如baseline，因此作为teacher没有意义

Funaudio:
✅ Overall CER: 0.0540
✅ Overall WER: 0.0798,指标上面都有很大的提升。可以作为teacher

我们进一步进行实验，对音频数据去混响，降噪，自动增益
为了获得更准确的伪标签。

✅ Enhanced CER: 0.0559
✅ Enhanced WER: 0.0798
我们尝试了基于 SNR 的音频增强策略，但发现整体 CER/WER 反而上升。分析表明，传统信号处理方法（如谱减法）引入的人工伪影对端到端 ASR 模型有害.
不降噪，只做音量归一化，效果没变

# Quality
## Audio Quality
使用UTMOS测评语音质量
越高越好（5 = 优秀，1 = 极差）
UTMOS 是 语言无关（language-independent） 的，因为它：

不依赖 ASR 或文本
只分析 声学特征（频谱、失真、噪声等）
在多语言数据（包括中文）上训练过

@inproceedings{baba2024utmosv2,
  title     = {The T05 System for The {V}oice{MOS} {C}hallenge 2024: Transfer Learning from Deep Image Classifier to Naturalness {MOS} Prediction of High-Quality Synthetic Speech},
  author    = {Baba, Kaito and Nakata, Wataru and Saito, Yuki and Saruwatari, Hiroshi},
  booktitle = {IEEE Spoken Language Technology Workshop (SLT)},
  year      = {2024},
}

pip install git+https://github.com/sarulab-speech/UTMOSv2.git

/timm/tf_efficientnetv2_s.in21k_ft_in1k

## Text Quality
因此，我们引入 语言模型合理性（LM Plausibility） 作为文本侧质量信号，通过预训练语言模型（PLM）对 ASR 输出文本的对数概率（log probability） 进行打分，量化其语言合理性。该方法无需额外标注，仅依赖文本本身，可有效补充音频侧指标的不足。
我们采用 uer/gpt2-chinese-cluecorpussmall —— 一个在大规模中文语料（CLUECorpus）上预训练的 GPT-2 架构因果语言模型。该模型：

支持中文文本的自回归概率建模；
轻量级（117M 参数），适合批量推理；
使用 CER 对齐后的文本（hyp_cer 字段）：已去除标点、转为小写、无多余空格；
保留英文术语（如 monorepo, drizzle），因其在上下文中具有语义；