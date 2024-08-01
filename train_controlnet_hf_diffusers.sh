#!/bin/bash

# run
# accelerate config

# check with
# accelerate env

MODEL_DIR="PixArt-alpha/PixArt-XL-2-1024-MS"
OUTPUT_DIR="./output/pixart-controlnet-open-pose"
TRAINING_DATA_DIR="/workspace/open_pose_controlnet"
VALIDATION_IMAGES=("$TRAINING_DATA_DIR/validation_images/pose/1_pose.jpg" "$TRAINING_DATA_DIR/validation_images/pose/2_pose.jpg" "$TRAINING_DATA_DIR/validation_images/pose/3_pose.jpg" "$TRAINING_DATA_DIR/validation_images/pose/4_pose.jpg")
VALIDATION_PROMPTS=("Friends standing in front of a modern building" "Two friends chatting in a park" "A woman enjoying a beautiful sunny day" "Three people on a business meeting")

accelerate launch ./controlnet/train_pixart_controlnet_hf.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --train_data_dir=$TRAINING_DATA_DIR \
 --resolution=1024 \
 --num_train_epochs=3 \
 --learning_rate=1e-5 \
 --train_batch_size=2 \
 --gradient_accumulation_steps=4 \
 --report_to="wandb" \
 --tracker_project_name="pixart_pose_controlnet" \
 --seed=42 \
 --dataloader_num_workers=8 \
 --validation_image=$VALIDATION_IMAGES \
 --validation_prompt=$VALIDATION_PROMPTS \
#  --num_validation_images=1 \
#  --validation_steps=100 \
#  --max_train_samples=15000 \
#  --checkpointing_steps=500 \
#  --lr_scheduler="cosine" --lr_warmup_steps=0 \
