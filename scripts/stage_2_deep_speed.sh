
# export TORCHAUDIO_USE_BACKEND_DISPATCHER=0
# IMPORTANT: this is the training script for the original LLaVA, NOT FOR LLaVA V1.5!

# Uncomment and set the following variables correspondingly to run this script:

MODEL_PATH="./training_checkpoints_fsdp_offcial/stage_2/Qwen/Qwen2.5-1.5B/checkpoint-12000/cpu"
MODEL_VERSION=Qwen/Qwen2.5-1.5B
PROMPT_VERSION=qwen
########### DO NOT CHANGE ###########
# --include localhost:1 \
deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path $MODEL_PATH \
    --version $PROMPT_VERSION \
    --num_image_tokens 256 \
    --data_file_names speech_asr_debug.json  \
    --meta_json_path ./data/metadata_json_files  \
    --val_data_file_names speech_asr_debug.json,speech_vg_debug.json,speech_vqav2_debug.json,speech_laion_debug.json  \
    --val_meta_json_path ./data/metadata_json_files  \
    --image_folder ./data/vlm/images \
    --mm_projector_type mlp2x_gelu \
    --vision_tower openai/clip-vit-large-patch14 \
    --mm_vision_select_layer -2 \
    --bf16 true \
    --output_dir ./training_checkpoints_dp/stage_2/$MODEL_VERSION \
    --num_train_epochs 2 \
    --do_eval \
    --eval_strategy "steps" \
    --eval_steps 1000 \
    --save_strategy  "steps" \
    --save_steps 1000 \
    --per_device_train_batch_size 6 \
    --gradient_accumulation_steps 1 \
    --save_total_limit 0 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4000 \
    --gradient_checkpointing False \
    --dataloader_num_workers 4 \
    --lazy_preprocess True


    #  --data_file_names speech_tts.json,speech_vg.json,speech_vqav2.json,speech_asr.json,speech_laion_part_1.json,speech_laion_part_2.json,speech_laion_part_3.json \
