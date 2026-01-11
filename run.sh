echo "开始配置 LogLabel Booster 环境..."

#配置环境
pip install -r requirments.txt

#配置TQ
# 如果没有./uer/gpt2-chinese-cluecorpussmall路径就下载到本地根目录：
# brew install git-xet
# git xet install
# git clone git@hf.co:souljoy/gpt2-small-chinese-cluecorpussmall
TQ_MODEL_DIR="uer/gpt2-chinese-cluecorpussmall"
if [ ! -d "$TQ_MODEL_DIR" ]; then
    echo "下载 TQ 模型: $TQ_MODEL_DIR ..."
    # 尝试用 git-xet（HuggingFace 大模型）
    if command -v brew &> /dev/null; then
        brew install git-xet
        git xet install
    else
        echo "⚠️ brew not found, skipping git-xet (may affect clone speed)"
    fi
    git clone https://huggingface.co/souljoy/gpt2-small-chinese-cluecorpussmall "$TQ_MODEL_DIR"
else
    echo "TQ 模型已存在: $TQ_MODEL_DIR"
fi


#配置Teacher ASR
# 如果没有./Fun-ASR以及./FunAudioLLM路径就下载到本地根目录：
# modelscope download --model FunAudioLLM/Fun-ASR-Nano-2512  --local_dir ./

# git clone https://github.com/FunAudioLLM/Fun-ASR.git
ASR_MODEL_DIR="FunAudioLLM"
ASR_REPO_DIR="Fun-ASR"

if [ ! -d "$ASR_MODEL_DIR" ]; then
    echo "download Teacher ASR: FunAudioLLM/Fun-ASR-Nano-2512 ..."
    modelscope download --model FunAudioLLM/Fun-ASR-Nano-2512 --local_dir ./
else
    echo "ASR 模型已存在: $ASR_MODEL_DIR"
fi

if [ ! -d "$ASR_REPO_DIR" ]; then
    echo "clone Fun-ASR ..."
    git clone https://github.com/FunAudioLLM/Fun-ASR.git
    # pip install -r Fun-ASR/requirements.txt
else
    echo "Fun-ASR already exists: $ASR_REPO_DIR"
fi

#切分数据
echo "划分数据集 ..."
python data_split.py


##推理：
echo "infer"
# 处理 dev 集
echo "infer dev data"

python run.py \
  --input_dir data/ \
  --output_dir outputs/dev/ \
  --subset dev

# 处理 train_like（用于训练）
echo "infer train-like data"
python run.py \
  --input_dir data/ \
  --output_dir outputs/train_like/ \
  --subset train_like

# 处理 test（用于提交）
echo "infer test data"
python run.py \
  --input_dir interview_data/ \
  --output_dir outputs/test/ \
  --subset test