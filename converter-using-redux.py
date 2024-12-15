#!/usr/bin/env python3
import argparse
import os
import torch
from diffusers import FluxPriorReduxPipeline, FluxPipeline
from diffusers.utils import load_image
from tqdm import tqdm
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5Tokenizer

def process_images_with_flux_redux(img_folder_path: str, img_output_folder_path: str, batch_size: int = 1):
    """
    Process all images in a folder through FLUX Redux pipeline with D3VR4JPUT LoRA.
    
    Args:
        img_folder_path (str): Path to folder containing input images
        img_output_folder_path (str): Path to folder where processed images will be saved
        batch_size (int): Number of images to process in parallel (if supported by GPU memory)
    """
    # Create output directory if it doesn't exist
    os.makedirs(img_output_folder_path, exist_ok=True)
    
    # Initialize pipelines
    print("Initializing FLUX Redux pipelines...")
    clip = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", max_length=77, torch_dtype=torch.bfloat16).to("cuda")
    clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", max_length=77, torch_dtype=torch.bfloat16)
    t5 = T5EncoderModel.from_pretrained("google/t5-v1_1-xxl", max_length=512, torch_dtype=torch.bfloat16).to("cuda")
    t5_tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-xxl", max_length=512, torch_dtype=torch.bfloat16)
    pipe_prior_redux = FluxPriorReduxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Redux-dev",
        text_encoder = clip,
        text_encoder_2 = t5,
        tokenizer = clip_tokenizer,
        tokenizer_2 = t5_tokenizer,
        torch_dtype=torch.bfloat16
    ).to("cuda")
    
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        text_encoder=None,
        text_encoder_2=None,
        torch_dtype=torch.bfloat16
    ).to("cuda")
    
    # Load LoRA weights
    pipe.load_lora_weights("thesantatitan/devrajput")
    
    # Get list of image files
    valid_extensions = {'.png', '.jpg', '.jpeg', '.webp'}
    image_files = [
        f for f in os.listdir(img_folder_path)
        if os.path.splitext(f.lower())[1] in valid_extensions
    ]
    
    
    if not image_files:
        print(f"No valid images found in {img_folder_path}")
        return
    
    print(f"Found {len(image_files)} images to process")
    image_files.sort()
    # Process each image with progress bar
    for image_file in tqdm(image_files, desc="Processing images"):
        try:
            # Load input image
            input_path = os.path.join(img_folder_path, image_file)
            input_image = load_image(input_path)
            
            # Get original dimensions
            width, height = input_image.size
            
            # Process through prior redux pipeline
            pipe_prior_output = pipe_prior_redux(input_image, prompt="D3VR4JPUT")
            
            # Generate final image
            images = pipe(
                guidance_scale=20,
                num_inference_steps=7,
                generator=torch.Generator("cpu").manual_seed(42),
                width=width,
                height=height,
                **pipe_prior_output,
            ).images
            
            # Save output image with same name
            output_path = os.path.join(img_output_folder_path, image_file)
            images[0].save(output_path)
            
        except Exception as e:
            print(f"\nError processing {image_file}: {str(e)}")
            continue
    
    print(f"\nProcessing complete! Successfully processed {len(image_files)} images.")
    
    # Clear GPU memory
    del pipe_prior_redux
    del pipe
    torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser(description="FLUX Redux Image Processing CLI")
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Input folder containing images to process"
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output folder for processed images"
    )
    parser.add_argument(
        "-b", "--batch-size",
        type=int,
        default=1,
        help="Batch size for processing (default: 1)"
    )
    parser.add_argument(
        "--device",
        choices=['cuda', 'cpu'],
        default='cuda',
        help="Device to run the model on (default: cuda)"
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=2.5,
        help="Guidance scale for generation (default: 2.5)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of inference steps (default: 50)"
    )

    args = parser.parse_args()

    # Validate input folder
    if not os.path.exists(args.input):
        print(f"Error: Input folder '{args.input}' does not exist!")
        return

    if not os.path.isdir(args.input):
        print(f"Error: '{args.input}' is not a directory!")
        return

    # Check CUDA availability if cuda is selected
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA selected but not available. Defaulting to CPU.")
        args.device = 'cpu'

    try:
        process_images_with_flux_redux(args.input, args.output, args.batch_size)
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user.")
        # Clear GPU memory on interrupt
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        # Clear GPU memory on error
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()