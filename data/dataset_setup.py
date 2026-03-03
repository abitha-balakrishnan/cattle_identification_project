"""
Dataset Setup Script for Indian Cattle & Buffalo Breed Recognition

This script creates the required directory structure for the dataset,
validates image formats, and provides utilities for dataset preparation.
"""

import os
import shutil
import random
from pathlib import Path
from typing import List, Tuple
import argparse

# Define the target Indian cattle and buffalo breeds
INDIAN_CATTLE_BREEDS = [
    "Gir", "Sahiwal", "Ongole", "Red_Sindhi", "Tharparkar", 
    "Kankrej", "Hariana", "Rathi"
]

INDIAN_BUFFALO_BREEDS = [
    "Murrah", "Jaffarabadi", "Surti", "Mehsana", "Nili_Ravi", "Pandharpuri"
]

ALL_BREEDS = INDIAN_CATTLE_BREEDS + INDIAN_BUFFALO_BREEDS


def create_dataset_structure(base_path: str = "./dataset") -> None:
    """
    Creates the required dataset directory structure:
    dataset/
    ├── train/
    ├── val/
    └── test/
        └── [breed folders]
    """
    base_path = Path(base_path)
    
    # Create main dataset directories
    splits = ['train', 'val', 'test']
    for split in splits:
        split_path = base_path / split
        split_path.mkdir(parents=True, exist_ok=True)
        
        # Create breed subdirectories
        for breed in ALL_BREEDS:
            breed_path = split_path / breed
            breed_path.mkdir(exist_ok=True)
    
    print(f"Dataset structure created at: {base_path}")
    print(f"Created {len(splits)} splits with {len(ALL_BREEDS)} breeds each")


def validate_image_file(file_path: Path) -> bool:
    """
    Validates if a file is a valid image format.
    """
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    return file_path.suffix.lower() in valid_extensions


def split_dataset(source_dir: str, dest_dir: str, train_ratio: float = 0.7, 
                  val_ratio: float = 0.2, test_ratio: float = 0.1) -> None:
    """
    Splits a source dataset into train/val/test sets based on ratios.
    """
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)
    
    if not source_path.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_path}")
    
    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
    
    # Create destination structure
    create_dataset_structure(dest_path)
    
    # Process each breed folder in source
    for breed_folder in source_path.iterdir():
        if not breed_folder.is_dir():
            continue
            
        breed_name = breed_folder.name
        
        # Get all valid image files
        image_files = []
        for img_file in breed_folder.iterdir():
            if validate_image_file(img_file):
                image_files.append(img_file)
        
        if not image_files:
            print(f"Warning: No valid images found in {breed_folder}")
            continue
        
        # Shuffle the files randomly
        random.shuffle(image_files)
        
        # Calculate split indices
        total_count = len(image_files)
        train_end = int(train_ratio * total_count)
        val_end = train_end + int(val_ratio * total_count)
        
        # Split the files
        train_files = image_files[:train_end]
        val_files = image_files[train_end:val_end]
        test_files = image_files[val_end:]
        
        # Copy files to respective directories
        for split_name, files in [("train", train_files), ("val", val_files), ("test", test_files)]:
            split_dest = dest_path / split_name / breed_name
            for img_file in files:
                dest_file = split_dest / img_file.name
                shutil.copy2(img_file, dest_file)
        
        print(f"Split {breed_name}: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")


def get_dataset_stats(dataset_path: str) -> dict:
    """
    Gets statistics about the dataset.
    """
    dataset_path = Path(dataset_path)
    stats = {}
    
    for split in ['train', 'val', 'test']:
        split_path = dataset_path / split
        if not split_path.exists():
            continue
            
        split_stats = {}
        total_split_images = 0
        
        for breed in ALL_BREEDS:
            breed_path = split_path / breed
            if breed_path.exists():
                image_count = len([f for f in breed_path.iterdir() if validate_image_file(f)])
                split_stats[breed] = image_count
                total_split_images += image_count
        
        stats[split] = {
            'breeds': split_stats,
            'total': total_split_images
        }
    
    return stats


def print_dataset_report(stats: dict) -> None:
    """
    Prints a formatted dataset report.
    """
    print("\n" + "="*60)
    print("DATASET REPORT")
    print("="*60)
    
    total_images = 0
    for split, split_data in stats.items():
        print(f"\n{split.upper()} SET:")
        print("-" * 20)
        
        split_total = 0
        for breed, count in split_data['breeds'].items():
            print(f"  {breed:<15}: {count:>4} images")
            split_total += count
        
        print(f"  {'TOTAL':<15}: {split_total:>4} images")
        total_images += split_total
    
    print(f"\nGRAND TOTAL: {total_images} images")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Dataset setup utility for cattle breed recognition")
    parser.add_argument("--setup", action="store_true", help="Create dataset directory structure")
    parser.add_argument("--source", type=str, help="Source directory for dataset splitting")
    parser.add_argument("--dest", type=str, default="./dataset", help="Destination directory")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Training ratio (default: 0.7)")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio (default: 0.2)")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test ratio (default: 0.1)")
    parser.add_argument("--report", action="store_true", help="Generate dataset report")
    
    args = parser.parse_args()
    
    if args.setup:
        create_dataset_structure(args.dest)
        print(f"Dataset structure created at {args.dest}")
    
    if args.source:
        split_dataset(
            args.source, args.dest,
            args.train_ratio, args.val_ratio, args.test_ratio
        )
        print(f"Dataset split from {args.source} to {args.dest}")
    
    if args.report:
        stats = get_dataset_stats(args.dest)
        print_dataset_report(stats)


if __name__ == "__main__":
    main()