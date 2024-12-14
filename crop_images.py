#!/usr/bin/env python3

import argparse
import os
import sys
import cv2
import numpy as np
from pathlib import Path

def crop_images_with_masks(img_folder_path, mask_folder_path, 
                         output_folder_path, output_mask_folder_path,
                         min_bbox_size=50):
    """
    Crop images based on their corresponding masks and save to output folders.
    Handles transparent PNG masks where the object is black and background is transparent.
    
    Args:
        img_folder_path (str): Path to folder containing original images (JPG)
        mask_folder_path (str): Path to folder containing mask images (PNG)
        output_folder_path (str): Path to save cropped images
        output_mask_folder_path (str): Path to save cropped masks
        min_bbox_size (int): Minimum bounding box size threshold (default: 50 pixels)
    """
    # Create output directories if they don't exist
    Path(output_folder_path).mkdir(parents=True, exist_ok=True)
    Path(output_mask_folder_path).mkdir(parents=True, exist_ok=True)
    
    # Get list of jpg image files
    img_files = [f for f in os.listdir(img_folder_path) 
                 if f.lower().endswith('.jpg')]
    
    total_files = len(img_files)
    processed_files = 0
    
    print(f"Found {total_files} images to process.")
    
    for img_file in img_files:
        try:
            # Construct paths with appropriate extensions
            img_path = os.path.join(img_folder_path, img_file)
            mask_file = os.path.splitext(img_file)[0] + '.png'
            mask_path = os.path.join(mask_folder_path, mask_file)
            
            if not os.path.exists(mask_path):
                print(f"Warning: No corresponding mask found for {img_file} (looking for {mask_file})")
                continue
                
            img = cv2.imread(img_path)
            # Read mask with alpha channel
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            
            if img is None or mask is None:
                print(f"Error: Could not read image or mask for {img_file}")
                continue
            
            # Extract alpha channel if it exists
            if mask.shape[-1] == 4:  # If mask has alpha channel
                # Get alpha channel (255 for opaque black, 0 for transparent)
                alpha_channel = mask[:, :, 3]
                # Create binary mask (255 where opaque, 0 where transparent)
                binary_mask = np.where(alpha_channel > 0, 255, 0).astype(np.uint8)
            else:
                binary_mask = mask
            
            # Find contours in the binary mask
            contours, _ = cv2.findContours(binary_mask, 
                                         cv2.RETR_EXTERNAL, 
                                         cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                print(f"Warning: No contours found in mask for {img_file}")
                # Save original image and mask as is, maintaining their original formats
                cv2.imwrite(os.path.join(output_folder_path, img_file), img)
                cv2.imwrite(os.path.join(output_mask_folder_path, mask_file), mask)
                continue
            
            # Find the bounding box that contains all contours
            x_min = y_min = float('inf')
            x_max = y_max = 0
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x + w)
                y_max = max(y_max, y + h)
            
            # Calculate bounding box dimensions
            bbox_width = x_max - x_min
            bbox_height = y_max - y_min
            
            # Check if bounding box is too small
            if bbox_width < min_bbox_size or bbox_height < min_bbox_size:
                print(f"Warning: Bounding box too small for {img_file}, saving original")
                cv2.imwrite(os.path.join(output_folder_path, img_file), img)
                cv2.imwrite(os.path.join(output_mask_folder_path, mask_file), mask)
                continue
            
            # Crop image and mask
            cropped_img = img[y_min:y_max, x_min:x_max]
            cropped_mask = mask[y_min:y_max, x_min:x_max]
            
            # Save cropped images with their respective formats
            cv2.imwrite(os.path.join(output_folder_path, img_file), cropped_img)
            cv2.imwrite(os.path.join(output_mask_folder_path, mask_file), cropped_mask)
            
            processed_files += 1
            print(f"Processed {img_file} ({processed_files}/{total_files})")
            
        except Exception as e:
            print(f"Error processing {img_file}: {str(e)}")
            continue
    
    print(f"\nProcessing complete. Successfully processed {processed_files} out of {total_files} images.")

def validate_directory(path):
    """Validate if directory exists."""
    if not os.path.isdir(path):
        raise argparse.ArgumentTypeError(f"'{path}' is not a valid directory")
    return path

def main():
    parser = argparse.ArgumentParser(
        description='Crop images based on their corresponding masks. Handles transparent PNG masks.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--images',
        type=validate_directory,
        required=True,
        help='Path to folder containing original images (JPG format)'
    )
    
    parser.add_argument(
        '--masks',
        type=validate_directory,
        required=True,
        help='Path to folder containing mask images (PNG format with transparency)'
    )
    
    parser.add_argument(
        '--output-images',
        type=str,
        required=True,
        help='Path to save cropped images (will save as JPG)'
    )
    
    parser.add_argument(
        '--output-masks',
        type=str,
        required=True,
        help='Path to save cropped masks (will save as PNG)'
    )
    
    parser.add_argument(
        '--min-size',
        type=int,
        default=50,
        help='Minimum bounding box size threshold in pixels'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed progress information'
    )
    
    args = parser.parse_args()
    
    try:
        print("Starting image processing...")
        crop_images_with_masks(
            args.images,
            args.masks,
            args.output_images,
            args.output_masks,
            args.min_size
        )
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()