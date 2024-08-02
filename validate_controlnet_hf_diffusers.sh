#!/bin/bash

# in mixed-precision mode, the the training script cannot validate the model, so we need to run the validation script separately

CHECKPOINTS_FOLDER="output/pixart-controlnet-hf-diffusers-test"
OUTPUT_DIR="output/controlnet/validation_controlnet_hf_diffusers"

CONTROL_IMAGES_FOLDER="output/controlnet/control_images"
CONTROL_IMAGE_1="$CONTROL_IMAGES_FOLDER/conditioning_image_1.png"
CONTROL_IMAGE_2="$CONTROL_IMAGES_FOLDER/conditioning_image_1.png"

VALIDATION_PROPMTP_1="red circle with blue background"
VALIDATION_PROPMTP_2="cyan circle with brown floral background"

python ./controlnet/validate_folder_with_checkpoints.py \
 --checkpoints_folder "$CHECKPOINTS_FOLDER" \
 --output_folder "$OUTPUT_DIR" \
 --control_images "$CONTROL_IMAGE_1" "$CONTROL_IMAGE_2" \
 --validation_prompts "$VALIDATION_PROPMTP_1" "$VALIDATION_PROPMTP_2"