#!/bin/bash
# export TORCHAUDIO_USE_BACKEND_DISPATCHER=0
# IMPORTANT: this is the training script for the original LLaVA, NOT FOR LLaVA V1.5!
# Uncomment and set the following variables correspondingly to run this script:
MODEL_PATH=Qwen/Qwen2.5-3B
MODEL_VERSION=Qwen/Qwen2.5-3B

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
    --num_image_tokens 256 \
    --train_metadata_names metadata_laion.json,metadata_vqav2.json,metadata_vg.json,metadata_tts.json,metadata_asr.json \
    --metadata_folder conversation_metadata \
    --data_path json_files \
    --text_output_ins True \
    --image_folder ./data/vlm/images \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end True \
    --mm_use_im_patch_token True \
    --vision_tower openai/clip-vit-large-patch14 \
    --bf16 true \
    --output_dir ./training_checkpoints_ddp_text_out_ins/stage_2/$MODEL_VERSION \
    --num_train_epochs 2 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --save_strategy  "steps" \
    --save_steps 50000 \
    --save_total_limit 0 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing False \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none