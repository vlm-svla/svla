#!/bin/bash
# export TORCHAUDIO_USE_BACKEND_DISPATCHER=0
# IMPORTANT: this is the training script for the original LLaVA, NOT FOR LLaVA V1.5!
export FSDP_CPU_RAM_EFFICIENT_LOADING=1
# Uncomment and set the following variables correspondingly to run this script:
MODEL_PATH=Qwen/Qwen2.5-7B
MODEL_VERSION=Qwen/Qwen2.5-7B

PROMPT_VERSION=qwen

# FSDP related configuration
NNODE=1
GPUS_PER_NODE=8

torchrun \
    --nnode $NNODE \
    --nproc_per_node $GPUS_PER_NODE \
    --master_addr 127.0.0.1 \
    --master_port 29501 \
    llava/train/train_mem.py \
    --model_name_or_path $MODEL_PATH \
    --version $PROMPT_VERSION \
    --data_file_names speech_asr_debug.json \
    --meta_json_path ./data/metadata_json_files \
    --image_folder ./data/vlm/images \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 true \
    --output_dir ./training_checkpoints/stage_1/$MODEL_VERSION \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --save_strategy  "steps" \
    --save_steps 20000 \
    --save_total_limit 0 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing False \
    --dataloader_num_workers 0 \
    --lazy_preprocess True \
    --report_to none \