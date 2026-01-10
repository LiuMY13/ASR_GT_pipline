brew install git-xet
git xet install

git clone git@hf.co:souljoy/gpt2-small-chinese-cluecorpussmall

modelscope download --model FunAudioLLM/Fun-ASR-Nano-2512  --local_dir ./

git clone https://github.com/FunAudioLLM/Fun-ASR.git

python data_split.py

python Merge_full_features.py

python analyze.py