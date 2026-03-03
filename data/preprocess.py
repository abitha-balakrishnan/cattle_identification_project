"""
Data Preprocessing Script for Indian Cattle & Buffalo Breed Recognition

This script handles image preprocessing tasks including resizing, normalization,
augmentation, and quality filtering for the cattle breed dataset.
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
from pathlib import Path
from typing import List, Tuple, Generator
import argparse


def is_blurry(image_path: str, threshold: float = 100.0) -> bool:
    """
    Detects if an image is blurry using Laplacian variance.
    Lower values indicate blurrier images.
    """
    image = cv2.imread(str(image_path))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold


def is_low_contrast(image_path: str, threshold: float = 0.15) -> bool:
    """
    Detects if an image has low contrast.
    """
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    hist = image.histogram()
    hist = [i / sum(hist) for i in hist]  # Normalize histogram
    
    # Calculate entropy as a measure of contrast
    entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in hist)
    return entropy < threshold


def resize_and_normalize(image_path: str, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Resizes and normalizes an image to the target size.
    """
    image = Image.open(image_path).convert('RGB')
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    
    # Convert to numpy array and normalize to [0, 1]
    image_array = np.array(image).astype(np.float32) / 255.0
    
    return image_array


def create_augmentation_pipeline():
    """
    Creates an augmentation pipeline for training data.
    """
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
        A.OneOf([
            A.GaussNoise(var_limit=(0.001, 0.005), p=0.3),
            A.MotionBlur(blur_limit=3, p=0.2),
        ], p=0.2),
        A.Resize(height=224, width=224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    val_transform = A.Compose([
        A.Resize(height=224, width=224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    return train_transform, val_transform


def validate_image_quality(image_path: str) -> Tuple[bool, List[str]]:
    """
    Validates image quality based on multiple criteria.
    Returns (is_valid, list_of_issues).
    """
    issues = []
    
    # Check if file exists and is readable
    try:
        img = Image.open(image_path)
        img.verify()
    except Exception as e:
        issues.append(f"Cannot read image: {str(e)}")
        return False, issues
    
    # Reopen image after verification
    img = Image.open(image_path)
    
    # Check dimensions (should not be too small)
    width, height = img.size
    if width < 100 or height < 100:
        issues.append(f"Image too small: {width}x{height}")
    
    # Check if blurry
    if is_blurry(image_path):
        issues.append("Image appears blurry")
    
    # Check contrast
    if is_low_contrast(image_path):
        issues.append("Image has low contrast")
    
    # Check file size (too small might indicate corrupted image)
    file_size = os.path.getsize(image_path)
    if file_size < 1024:  # Less than 1KB
        issues.append(f"File size too small: {file_size} bytes")
    
    return len(issues) == 0, issues


def preprocess_dataset(
    source_dir: str, 
    output_dir: str, 
    target_size: Tuple[int, int] = (224, 224),
    quality_check: bool = True
) -> dict:
    """
    Preprocesses the entire dataset.
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    stats = {
        'processed': 0,
        'skipped': 0,
        'quality_issues': 0,
        'total_errors': 0
    }
    
    # Create output directory structure
    for split in ['train', 'val', 'test']:
        split_source = source_path / split
        if not split_source.exists():
            continue
            
        split_output = output_path / split
        split_output.mkdir(parents=True, exist_ok=True)
        
        # Process each breed in the split
        for breed_dir in split_source.iterdir():
            if not breed_dir.is_dir():
                continue
                
            breed_output = split_output / breed_dir.name
            breed_output.mkdir(exist_ok=True)
            
            # Process each image in the breed directory
            for img_file in breed_dir.iterdir():
                if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']:
                    continue
                
                if quality_check:
                    is_valid, issues = validate_image_quality(img_file)
                    if not is_valid:
                        print(f"Quality issues with {img_file}: {', '.join(issues)}")
                        stats['quality_issues'] += 1
                        stats['skipped'] += 1
                        continue
                
                try:
                    # Resize and save the image
                    img = Image.open(img_file).convert('RGB')
                    img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
                    
                    output_file = breed_output / img_file.name
                    img_resized.save(output_file)
                    
                    stats['processed'] += 1
                    if stats['processed'] % 100 == 0:
                        print(f"Processed {stats['processed']} images...")
                        
                except Exception as e:
                    print(f"Error processing {img_file}: {str(e)}")
                    stats['total_errors'] += 1
                    stats['skipped'] += 1
    
    return stats


def generate_augmented_samples(
    source_dir: str, 
    output_dir: str, 
    augment_factor: int = 2
) -> None:
    """
    Generates augmented samples to balance the dataset.
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # Create augmentation transforms
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.OneOf([
            A.GaussNoise(var_limit=(0.001, 0.005), p=0.3),
            A.MotionBlur(blur_limit=3, p=0.2),
        ], p=0.3),
    ])
    
    for split in ['train', 'val']:  # Only augment train and val
        split_source = source_path / split
        if not split_source.exists():
            continue
            
        split_output = output_path / split
        split_output.mkdir(parents=True, exist_ok=True)
        
        for breed_dir in split_source.iterdir():
            if not breed_dir.is_dir():
                continue
                
            breed_output = split_output / breed_dir.name
            breed_output.mkdir(exist_ok=True)
            
            # Get original images
            original_images = [f for f in breed_dir.iterdir() 
                              if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
            
            for img_file in original_images:
                # Copy original image
                output_file = breed_output / img_file.name
                if not output_file.exists():  # Don't overwrite
                    img = Image.open(img_file).convert('RGB')
                    img.save(output_file)
                
                # Generate augmented versions
                img = np.array(Image.open(img_file).convert('RGB'))
                
                for i in range(augment_factor):
                    augmented = transform(image=img)['image']
                    aug_img = Image.fromarray(augmented)
                    
                    # Create augmented filename
                    stem = img_file.stem
                    suffix = img_file.suffix
                    aug_filename = f"{stem}_aug_{i}{suffix}"
                    aug_output_file = breed_output / aug_filename
                    aug_img.save(aug_output_file)


def calculate_dataset_statistics(dataset_path: str) -> dict:
    """
    Calculates and returns dataset statistics.
    """
    dataset_path = Path(dataset_path)
    stats = {}
    
    for split in ['train', 'val', 'test']:
        split_path = dataset_path / split
        if not split_path.exists():
            continue
            
        split_stats = {'total_images': 0, 'breeds': {}}
        
        for breed_dir in split_path.iterdir():
            if not breed_dir.is_dir():
                continue
                
            images = [f for f in breed_dir.iterdir() 
                     if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']]
            
            breed_count = len(images)
            split_stats['breeds'][breed_dir.name] = breed_count
            split_stats['total_images'] += breed_count
        
        stats[split] = split_stats
    
    return stats


def print_preprocessing_report(stats: dict) -> None:
    """
    Prints a formatted preprocessing report.
    """
    print("\n" + "="*60)
    print("PREPROCESSING REPORT")
    print("="*60)
    
    for split, split_stats in stats.items():
        print(f"\n{split.upper()} SPLIT:")
        print("-" * 30)
        print(f"Total images: {split_stats['total_images']}")
        
        if 'breeds' in split_stats:
            print("By breed:")
            for breed, count in split_stats['breeds'].items():
                print(f"  {breed:<20}: {count:>4} images")
    
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Data preprocessing for cattle breed recognition")
    parser.add_argument("--source", type=str, required=True, help="Source dataset directory")
    parser.add_argument("--output", type=str, required=True, help="Output directory for preprocessed data")
    parser.add_argument("--size", type=int, nargs=2, default=[224, 224], 
                       help="Target image size (height width), default: 224 224")
    parser.add_argument("--quality-check", action="store_true", 
                       help="Perform quality checks on images")
    parser.add_argument("--augment", action="store_true", 
                       help="Generate augmented samples")
    parser.add_argument("--augment-factor", type=int, default=2, 
                       help="Number of augmented samples per original image")
    parser.add_argument("--report", action="store_true", 
                       help="Generate dataset statistics report")
    
    args = parser.parse_args()
    
    if not args.report:
        print("Starting preprocessing...")
        stats = preprocess_dataset(
            args.source, 
            args.output, 
            tuple(args.size), 
            args.quality_check
        )
        print(f"Preprocessing completed. Stats: {stats}")
        
        if args.augment:
            print("Generating augmented samples...")
            generate_augmented_samples(
                args.output,
                args.output,
                args.augment_factor
            )
            print("Augmentation completed.")
    
    if args.report or args.augment:
        final_stats = calculate_dataset_statistics(args.output)
        print_preprocessing_report(final_stats)


if __name__ == "__main__":
    main()