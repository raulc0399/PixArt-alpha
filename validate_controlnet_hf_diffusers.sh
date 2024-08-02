#!/bin/bash

# in mixed-precision mode, the the training script cannot validate the model, so we need to run the validation script separately

CHECKPOINTS_FOLDER="output/pixart-controlnet-open-pose"
OUTPUT_DIR="output/controlnet/validation_controlnet_open_pose"

CONTROL_IMAGES_FOLDER="output/controlnet/control_images_pose"
CONTROL_IMAGE_1="$CONTROL_IMAGES_FOLDER/1_pose.png"
CONTROL_IMAGE_2="$CONTROL_IMAGES_FOLDER/2_pose.png"
CONTROL_IMAGE_3="$CONTROL_IMAGES_FOLDER/3_pose.png"
CONTROL_IMAGE_4="$CONTROL_IMAGES_FOLDER/4_pose.png"

VALIDATION_PROPMTP_1="Friends standing in front of a modern building"
VALIDATION_PROPMTP_2="Two friends chatting in a park"
VALIDATION_PROPMTP_3="A woman enjoying a beautiful sunny day"
VALIDATION_PROPMTP_4="Three people on a business meeting"

python ./controlnet/validate_folder_with_checkpoints.py \
 --checkpoints_folder "$CHECKPOINTS_FOLDER" \
 --output_folder "$OUTPUT_DIR" \
 --control_images "$CONTROL_IMAGE_1" "$CONTROL_IMAGE_2" "$CONTROL_IMAGE_3" "$CONTROL_IMAGE_4" \
 --validation_prompts "$VALIDATION_PROPMTP_1" "$VALIDATION_PROPMTP_2" "$VALIDATION_PROPMTP_3" "$VALIDATION_PROPMTP_4"