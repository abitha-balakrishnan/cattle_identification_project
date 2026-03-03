"""
Performance Analysis Script for Indian Cattle & Buffalo Breed Recognition

This script provides comprehensive analysis of model performance including
overfitting detection, class imbalance analysis, and recommendations for improvements.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import os
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize
import seaborn as sns
from tqdm import tqdm
import argparse
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


class CattleBreedDataset(Dataset):
    """Custom dataset for cattle and buffalo breed images."""
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # Get all breed classes
        self.classes = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        # Build image paths and labels
        self.samples = []
        for target_class in self.classes:
            class_index = self.class_to_idx[target_class]
            target_dir = self.data_dir / target_class
            for img_path in target_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']:
                    self.samples.append((img_path, class_index))
    
    def __getitem__(self, index):
        path, target = self.samples[index]
        image = Image.open(path).convert('RGB')
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image, target
    
    def __len__(self):
        return len(self.samples)


def load_trained_model(model_path, num_classes, model_name='resnet18'):
    """Loads a trained model."""
    
    if model_name == 'resnet18':
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'resnet50':
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    # Load the trained weights
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, checkpoint


def analyze_dataset_balance(data_dir, class_names):
    """Analyze class balance in the dataset."""
    print("Analyzing dataset balance...")
    
    class_counts = {}
    for class_name in class_names:
        class_path = Path(data_dir) / class_name
        if class_path.exists():
            count = len(list(class_path.glob('*')))
            class_counts[class_name] = count
    
    df = pd.DataFrame(list(class_counts.items()), columns=['Class', 'Count'])
    df = df.sort_values('Count', ascending=False)
    
    print("\nDataset Balance Summary:")
    print(df.to_string(index=False))
    
    # Calculate imbalance ratio
    max_count = df['Count'].max()
    min_count = df['Count'].min()
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    print(f"\nImbalance Ratio (max/min): {imbalance_ratio:.2f}")
    
    # Plot class distribution
    plt.figure(figsize=(12, 6))
    plt.bar(df['Class'], df['Count'])
    plt.title('Dataset Class Distribution')
    plt.xlabel('Breed')
    plt.ylabel('Number of Images')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('dataset_balance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return df, imbalance_ratio


def detect_overfitting(train_acc, val_acc, threshold=10):
    """Detects overfitting based on train-validation gap."""
    gap = train_acc - val_acc
    
    print(f"\nOverfitting Analysis:")
    print(f"Training Accuracy: {train_acc:.2f}%")
    print(f"Validation Accuracy: {val_acc:.2f}%")
    print(f"Gap (Train - Val): {gap:.2f}%")
    
    if gap > threshold:
        print(f"⚠️  WARNING: Significant overfitting detected (gap > {threshold}%)")
        recommendations = [
            "Increase dropout rate",
            "Add more data augmentation",
            "Use stronger regularization (higher weight decay)",
            "Reduce model complexity",
            "Early stopping",
            "Collect more data for underrepresented classes"
        ]
        print("Recommendations:")
        for rec in recommendations:
            print(f"  - {rec}")
        return True
    else:
        print(f"✅ No significant overfitting detected")
        return False


def analyze_class_performance(y_true, y_pred, class_names):
    """Analyze performance per class."""
    from sklearn.metrics import precision_recall_fscore_support
    
    precisions, recalls, f1_scores, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=range(len(class_names))
    )
    
    # Create DataFrame
    metrics_df = pd.DataFrame({
        'Class': class_names,
        'Precision': precisions,
        'Recall': recalls,
        'F1-Score': f1_scores
    })
    
    # Sort by F1-score to identify problematic classes
    metrics_df = metrics_df.sort_values('F1-Score', ascending=True)
    
    print("\nPer-Class Performance (sorted by F1-Score):")
    print(metrics_df.to_string(index=False))
    
    # Identify problematic classes (low F1-score)
    low_f1_classes = metrics_df[metrics_df['F1-Score'] < 0.7]['Class'].tolist()
    if low_f1_classes:
        print(f"\n⚠️  Classes with F1-Score < 0.7: {low_f1_classes}")
        print("Recommendations for these classes:")
        print("  - Collect more training data")
        print("  - Improve data quality")
        print("  - Consider data augmentation")
        print("  - Check for label accuracy")
    
    return metrics_df


def plot_confidence_distribution(y_true, y_pred, y_probs, class_names):
    """Plot prediction confidence distribution."""
    confidences = np.max(y_probs, axis=1)
    correct_predictions = (y_true == y_pred)
    
    plt.figure(figsize=(15, 5))
    
    # Overall confidence distribution
    plt.subplot(1, 3, 1)
    plt.hist(confidences, bins=50, alpha=0.7, edgecolor='black')
    plt.title('Overall Confidence Distribution')
    plt.xlabel('Confidence')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Confidence by correctness
    plt.subplot(1, 3, 2)
    correct_conf = confidences[correct_predictions]
    incorrect_conf = confidences[~correct_predictions]
    
    plt.hist(correct_conf, bins=30, alpha=0.7, label='Correct', color='green', density=True)
    plt.hist(incorrect_conf, bins=30, alpha=0.7, label='Incorrect', color='red', density=True)
    plt.title('Confidence by Correctness')
    plt.xlabel('Confidence')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Average confidence per class
    plt.subplot(1, 3, 3)
    avg_conf_per_class = []
    for i in range(len(class_names)):
        class_mask = (y_true == i)
        if np.sum(class_mask) > 0:
            avg_conf = np.mean(confidences[class_mask])
            avg_conf_per_class.append(avg_conf)
        else:
            avg_conf_per_class.append(0)
    
    plt.bar(range(len(class_names)), avg_conf_per_class)
    plt.title('Average Confidence per Class')
    plt.xlabel('Class Index')
    plt.ylabel('Average Confidence')
    plt.xticks(range(len(class_names)), [name[:10] + '...' if len(name) > 10 else name 
                                        for name in class_names], rotation=45)
    plt.tight_layout()
    
    plt.tight_layout()
    plt.savefig('confidence_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print confidence statistics
    print(f"\nConfidence Analysis:")
    print(f"Mean confidence: {np.mean(confidences):.3f}")
    print(f"Std confidence: {np.std(confidences):.3f}")
    print(f"Min confidence: {np.min(confidences):.3f}")
    print(f"Max confidence: {np.max(confidences):.3f}")
    print(f"Mean confidence (correct): {np.mean(correct_conf):.3f}")
    print(f"Mean confidence (incorrect): {np.mean(incorrect_conf):.3f}")


def analyze_misclassifications(y_true, y_pred, class_names):
    """Analyze patterns in misclassifications."""
    print("\nAnalyzing Misclassifications...")
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Find most commonly confused pairs
    misclassifications = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                misclassifications.append({
                    'true_class': class_names[i],
                    'predicted_class': class_names[j],
                    'count': cm[i, j]
                })
    
    # Sort by count
    misclassifications.sort(key=lambda x: x['count'], reverse=True)
    
    print("\nTop 10 Most Common Misclassification Pairs:")
    for i, miscl in enumerate(misclassifications[:10]):
        print(f"{i+1:2d}. {miscl['true_class']} → {miscl['predicted_class']}: {miscl['count']} times")
    
    return misclassifications


def plot_detailed_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plot a detailed confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names, 
        yticklabels=class_names
    )
    plt.title('Detailed Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_improvement_recommendations(
    imbalance_ratio, 
    overfitting_detected, 
    low_performance_classes,
    misclassifications
):
    """Generate comprehensive improvement recommendations."""
    print("\n" + "="*60)
    print("COMPREHENSIVE IMPROVEMENT RECOMMENDATIONS")
    print("="*60)
    
    recommendations = []
    
    # Dataset-related recommendations
    if imbalance_ratio > 3:
        recommendations.append("DATASET: Address severe class imbalance (ratio > 3:1)")
        recommendations.append("  - Use class weights during training")
        recommendations.append("  - Implement oversampling for minority classes")
        recommendations.append("  - Consider Synthetic Minority Oversampling Technique (SMOTE)")
    
    # Model-related recommendations
    if overfitting_detected:
        recommendations.append("MODEL: Address overfitting issues")
        recommendations.append("  - Increase dropout rate (try 0.5-0.7)")
        recommendations.append("  - Add stronger L2 regularization")
        recommendations.append("  - Implement early stopping")
        recommendations.append("  - Reduce model complexity")
        recommendations.append("  - Increase data augmentation")
    
    # Performance-related recommendations
    if low_performance_classes:
        recommendations.append("PERFORMANCE: Address low-performance classes")
        recommendations.append(f"  - Low F1-score classes: {low_performance_classes}")
        recommendations.append("  - Collect more diverse training examples for these classes")
        recommendations.append("  - Examine and clean mislabeled examples")
        recommendations.append("  - Consider ensemble methods for difficult classes")
    
    # Architecture-related recommendations
    recommendations.append("ARCHITECTURE: Consider model improvements")
    recommendations.append("  - Try different architectures (ResNet50, EfficientNet, Vision Transformer)")
    recommendations.append("  - Implement transfer learning from domain-specific models")
    recommendations.append("  - Use ensemble of multiple models")
    
    # Training-related recommendations
    recommendations.append("TRAINING: Optimize training process")
    recommendations.append("  - Use learning rate scheduling (cosine annealing, reduce on plateau)")
    recommendations.append("  - Implement gradient clipping")
    recommendations.append("  - Try different optimizers (AdamW, SGD with momentum)")
    recommendations.append("  - Use mixed precision training for faster convergence")
    
    # Data-related recommendations
    recommendations.append("DATA: Enhance data quality and quantity")
    recommendations.append("  - Apply advanced augmentation techniques")
    recommendations.append("  - Ensure consistent image preprocessing")
    recommendations.append("  - Remove mislabeled or low-quality images")
    recommendations.append("  - Collect images from diverse sources and conditions")
    
    # Print recommendations
    for i, rec in enumerate(recommendations, 1):
        print(f"{i:2d}. {rec}")
    
    print("="*60)


def comprehensive_performance_analysis(
    model_path,
    test_dir,
    model_name,
    output_dir='./performance_analysis'
):
    """Perform comprehensive performance analysis."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    test_dataset = CattleBreedDataset(Path(test_dir))
    num_classes = len(test_dataset.classes)
    class_names = test_dataset.classes
    
    model, checkpoint = load_trained_model(model_path, num_classes, model_name)
    model = model.to(device)
    
    print(f"Model loaded with {num_classes} classes")
    print(f"Classes: {class_names}")
    
    # Create test loader
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = CattleBreedDataset(Path(test_dir), transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Get predictions
    print("Generating predictions...")
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Predicting"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            
            _, preds = outputs.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_probs = np.array(all_probs)
    
    # Calculate overall accuracy
    accuracy = 100. * sum(np.array(all_preds) == np.array(all_targets)) / len(all_targets)
    print(f"\nOverall Test Accuracy: {accuracy:.2f}%")
    
    # 1. Dataset balance analysis
    class_counts_df, imbalance_ratio = analyze_dataset_balance(test_dir, class_names)
    
    # 2. Class performance analysis
    metrics_df = analyze_class_performance(all_targets, all_preds, class_names)
    low_performance_classes = metrics_df[metrics_df['F1-Score'] < 0.7]['Class'].tolist()
    
    # 3. Misclassification analysis
    misclassifications = analyze_misclassifications(all_targets, all_preds, class_names)
    
    # 4. Confidence analysis
    plot_confidence_distribution(all_targets, all_preds, all_probs, class_names)
    
    # 5. Detailed confusion matrix
    plot_detailed_confusion_matrix(all_targets, all_preds, class_names, 
                                 os.path.join(output_dir, 'detailed_confusion_matrix.png'))
    
    # 6. Generate improvement recommendations
    # For overfitting detection, we need training and validation accuracy
    # If not available, we'll assume no overfitting for now
    overfitting_detected = False  # This would be determined from training logs
    
    generate_improvement_recommendations(
        imbalance_ratio, 
        overfitting_detected, 
        low_performance_classes,
        misclassifications
    )
    
    # Save results
    results = {
        'overall_accuracy': accuracy,
        'dataset_imbalance_ratio': imbalance_ratio,
        'low_performance_classes': low_performance_classes,
        'total_misclassifications': len([1 for t, p in zip(all_targets, all_preds) if t != p]),
        'class_metrics': metrics_df.to_dict('records'),
        'top_misclassifications': misclassifications[:10]
    }
    
    with open(os.path.join(output_dir, 'performance_analysis.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nAnalysis completed! Results saved to: {output_dir}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Comprehensive model performance analysis")
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to trained model checkpoint")
    parser.add_argument("--test-dir", type=str, required=True,
                       help="Path to test dataset directory")
    parser.add_argument("--model-name", type=str, default="resnet18",
                       choices=["resnet18", "resnet50", "mobilenet_v2", "efficientnet_b0"],
                       help="Model architecture")
    parser.add_argument("--output-dir", type=str, default="./performance_analysis",
                       help="Directory to save analysis results")
    
    args = parser.parse_args()
    
    results = comprehensive_performance_analysis(
        model_path=args.model_path,
        test_dir=args.test_dir,
        model_name=args.model_name,
        output_dir=args.output_dir
    )
    
    print(f"\nPerformance analysis completed successfully!")
    print(f"Results saved in: {args.output_dir}")


if __name__ == "__main__":
    main()