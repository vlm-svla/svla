#!/bin/bash
# export TORCHAUDIO_USE_BACKEND_DISPATCHER=0
# IMPORTANT: this is the training script for the original LLaVA, NOT FOR LLaVA V1.5!

# Uncomment and set the following variables correspondingly to run this script:

MODEL_PATH=Qwen/Qwen2.5-7B
MODEL_VERSION=Qwen/Qwen2.5-7B
MAX_AUDIO_DURATION=15
AUDIO_TOKENIZER=speech_tokenizer

PROMPT_VERSION=qwen
########### DO NOT CHANGE ###########
# --include localhost:1 \
deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path $MODEL_PATH \
    --version $PROMPT_VERSION \
    --data_path "./data/jsons/libriheavy.json, ./data/jsons/squad_ss.json" \
    --audio_folder ./data/audio/ \
    --bf16 true \
    --audio_tokenizer $AUDIO_TOKENIZER \
    --max_audio_duration $MAX_AUDIO_DURATION \
    --output_dir ./training_checkpoints/stage_1/$MODEL_VERSION-$AUDIO_TOKENIZER-$MAX_AUDIO_DURATION \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 0 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4000 \
    --gradient_checkpointing False \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none
