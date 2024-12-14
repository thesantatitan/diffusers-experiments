#!/usr/bin/env python3
import argparse
import os
import torch
from diffusers import FluxControlPipeline, FluxTransformer2DModel
from diffusers.utils import load_image
from image_gen_aux import DepthPreprocessor
from tqdm import tqdm
from PIL import Image

def process_images_with_flux(img_folder_path: str, img_output_folder_path: str, batch_size: int = 1):
    """
    Process all images in a folder through FLUX depth pipeline with D3VR4JPUT LoRA.
    
    Args:
        img_folder_path (str): Path to folder containing input images
        img_output_folder_path (str): Path to folder where processed images will be saved
        batch_size (int): Number of images to process in parallel (if supported by GPU memory)
    """
    # Create output directory if it doesn't exist
    os.makedirs(img_output_folder_path, exist_ok=True)
    
    # Initialize pipeline
    print("Initializing FLUX pipeline...")
    pipe = FluxControlPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Depth-dev",
        torch_dtype=torch.bfloat16
    )
    pipe.load_lora_weights("thesantatitan/devrajput")
    pipe.to(device="cuda")
    
    # Initialize depth processor
    processor = DepthPreprocessor.from_pretrained("LiheYoung/depth-anything-large-hf")
    
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
    
    # Process each image with progress bar
    for image_file in tqdm(image_files, desc="Processing images"):
        try:
            # Load and process input image
            input_path = os.path.join(img_folder_path, image_file)
            control_image = load_image(input_path)
            
            # Get original dimensions
            width, height = control_image.size
            
            # Process through depth preprocessor
            control_image = processor(control_image)[0].convert("RGB")
            
            # Generate image using original dimensions
            prompt = "D3VR4JPUT"
            image = pipe(
                prompt=prompt,
                control_image=control_image,
                height=height,
                width=width,
                num_inference_steps=10,
                guidance_scale=4.0,
                generator=torch.Generator().manual_seed(42),
            ).images[0]
            
            # Save output image with same name
            output_path = os.path.join(img_output_folder_path, image_file)
            image.save(output_path)
            
        except Exception as e:
            print(f"\nError processing {image_file}: {str(e)}")
            continue
    
    print(f"\nProcessing complete! Successfully processed {len(image_files)} images.")

def main():
    parser = argparse.ArgumentParser(description="FLUX Image Processing CLI")
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
        process_images_with_flux(args.input, args.output, args.batch_size)
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user.")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")

if __name__ == "__main__":
    main()