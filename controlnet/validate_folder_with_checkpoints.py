import os
import sys
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
import argparse

script_dir = os.path.dirname(os.path.abspath(__file__))
subfolder_path = os.path.join(script_dir, 'pipeline')
sys.path.insert(0, subfolder_path)

from pipeline.pixart_controlnet_transformer import PixArtControlNetAdapterModel
from pipeline.pipeline_pixart_alpha_controlnet import PixArtAlphaControlnetPipeline, get_closest_hw

# MODEL_ID="PixArt-alpha/PixArt-XL-2-1024-MS"
MODEL_ID="PixArt-alpha/PixArt-XL-2-512x512"

def process_checkpoint_folder(checkpoint_folder, output_folder, checkpoint_number, prompts, validation_images, image_size, weight_dtype, device):
    controlnet = PixArtControlNetAdapterModel.from_pretrained(
        checkpoint_folder,
        torch_dtype=weight_dtype,
        use_safetensors=True,
    ).to(device)

    pipe = PixArtAlphaControlnetPipeline.from_pretrained(
        MODEL_ID,
        controlnet=controlnet,
        torch_dtype=weight_dtype,
        use_safetensors=True,
    ).to(device)

    for i, prompt in enumerate(prompts):
        with torch.no_grad():
            out = pipe(
                prompt=prompt,
                image=validation_images[i],
                num_inference_steps=14,
                guidance_scale=4.5,
                height=image_size,
                width=image_size,
            )
            
            checkpoint_number_str = str(checkpoint_number).zfill(8)
            output_image_path = os.path.join(output_folder, f"{checkpoint_number_str}_img_{i+1}.jpg")
            out.images[0].save(output_image_path)

            print(f"\tSaved image to {output_image_path}")

    # Clean up models
    del controlnet
    del pipe

    # Clean up GPU memory
    torch.cuda.empty_cache()

    print(f"\033[32mFinished processing checkpoint {checkpoint_folder}!\033[0m")

def generate_images_from_checkpoints(checkpoints_folder, output_folder, prompts, control_images, image_size=1024, weight_dtype=torch.float16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    validation_images = []
    for control_image in control_images:
        validation_image = Image.open(control_image).convert("RGB")
        validation_image = validation_image.resize((image_size, image_size))
        validation_images.append(validation_image)

    if len(validation_images) == 0:
        print("Validation images are empty.")
        return

    print(f"Validation images: {len(validation_images)}")

    print(f"Checking folder for checkpoints {checkpoints_folder}")
    for folder in os.listdir(checkpoints_folder):
        checkpoint_folder = os.path.join(checkpoints_folder, folder, "controlnet")

        if os.path.isdir(checkpoint_folder):
            print(f"\033[33mFound checkpoint from {checkpoint_folder}\033[0m")

            checkpoint_number = os.path.basename(folder).split('-')[-1]
            process_checkpoint_folder(checkpoint_folder, output_folder, checkpoint_number, prompts, validation_images, image_size, weight_dtype, device)

    # also try the controlnet subfolder directly
    checkpoint_folder = os.path.join(checkpoints_folder, "controlnet")
    if os.path.isdir(checkpoint_folder):
            print(f"\033[33mFound checkpoint from {checkpoint_folder}\033[0m")

            checkpoint_number = "final"
            process_checkpoint_folder(checkpoint_folder, output_folder, checkpoint_number, prompts, validation_images, image_size, weight_dtype, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script to validate folder with checkpoints.')
    parser.add_argument('--checkpoints_folder', type=str, help='Path to the folder containing checkpoints', required=True)
    parser.add_argument('--output_folder', type=str, help='Path to the output folder', required=True)
    parser.add_argument('--control_images', nargs='+', help='Path to the folder containing control images', required=True)
    parser.add_argument('--validation_prompts', nargs='+', help='List of validation prompts', required=True)
    args = parser.parse_args()

    checkpoints_folder = args.checkpoints_folder
    output_folder = args.output_folder
    control_images = args.control_images
    prompts = args.validation_prompts

    assert len(prompts) == len(control_images)

    # Check if the output folder exists, if not, create it
    if not os.path.exists(output_folder):
        print(f"Creating output folder {output_folder}")
        os.makedirs(output_folder)

    generate_images_from_checkpoints(checkpoints_folder, output_folder, prompts, control_images, image_size=512, weight_dtype=torch.float16)

