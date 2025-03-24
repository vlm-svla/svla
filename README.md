# 🌋 SVLA: A Unified Speech-Text-Vision Assistant with Multimodal Reasoning and Speech Generation

## Contents
- [Install](#install)
- [SVLA Weights](#svla-weights)
- [Demo](#Demo)
- [Dataset](#dataset)
- [Train](#train)
- [Evaluation](#evaluation)

## Install
This code is built from [LLaVA](https://github.com/haotian-liu/LLaVA)

1. Clone this repository and navigate to svla folder
```bash
git clone https://github.com/vlm-svla/svla.git
cd svla
```

2. Install Package
```Shell
conda create -n svla python=3.10 -y
conda activate svla
pip install --upgrade pip
pip install -e .
```

3. Install additional packages for training cases
```
pip install -e ".[train]"
# -----------------------------------------
# add these if you face some issues when installing flash-attention
# export PATH=/usr/local/cuda/bin:$PATH
# export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
# -----------------------------------------
pip install flash-attn==2.7.2.post1 --no-build-isolation --no-cache-dir
```

### Install TTS for inferent
You need to install [MeloTTS](https://github.com/myshell-ai/MeloTTS) for inference. We edit TTS to suitable with our environment. You can install by following:

```bash
cd MeloTTS
pip install -e .
python -m unidic download
cd ../
```


## LLaVA Weights
To use SVLA, download the required model weights:
```bash
mkdir weights
wget  -P ./weights/ https://huggingface.co/svla-vlm/svla/resolve/main/svla-sft-text-ins.zip 
unzip ./weights/svla-sft-text-ins.zip -d ./weights/
wget -P ./weights/ https://huggingface.co/fnlp/AnyGPT-speech-modules/resolve/main/speechtokenizer/ckpt.dev 
wget -P ./weights/ https://huggingface.co/fnlp/AnyGPT-speech-modules/resolve/main/speechtokenizer/config.json
wget -P ./weights/ https://huggingface.co/fnlp/AnyGPT-speech-modules/resolve/main/soundstorm/speechtokenizer_soundstorm_mls.pt
```

## Demo
We are currently developing a Gradio demo.
In the meantime, you can try a simple demo using:
```python
python inference_vlm.py
```
### Dataset
The datasets for Stage 2 and Stage 3 are available here:

👉 [https://huggingface.co/datasets/svla-vlm/svla](https://huggingface.co/datasets/svla-vlm/svla)

**Notes:**
- The provided datasets consist of tokenized speech represented as discrete tokens.
- In the paper, we prefix the speech tokens with `<speech_i>`, but in the dataset, the prefix is `<audio_i>`.


## Train
Comming soon



## Acknowledgement

- [LLava](https://github.com/haotian-liu/LLaVA): the codebase we built upon, and our base model Llava that has the amazing language capabilities!
- [MeloTTS](https://github.com/myshell-ai/MeloTTS)
- [SpeechTokenizer](https://github.com/myshell-ai/MeloTTS)# svla
