"""
Model Evaluation Script for Indian Cattle & Buffalo Breed Recognition

This script evaluates the trained model using various metrics and provides
detailed analysis of model performance.
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
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import seaborn as sns
from tqdm import tqdm
import argparse
import pandas as pd


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


def calculate_metrics_per_class(y_true, y_pred, class_names):
    """Calculates precision, recall, and F1-score for each class."""
    from sklearn.metrics import precision_recall_fscore_support
    
    precisions, recalls, f1_scores, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None
    )
    
    metrics_df = pd.DataFrame({
        'Class': class_names,
        'Precision': precisions,
        'Recall': recalls,
        'F1-Score': f1_scores
    })
    
    return metrics_df


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plots the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return cm


def plot_roc_curves_multiclass(y_true, y_scores, class_names, save_path):
    """Plots ROC curves for multiclass classification."""
    # Binarize the output
    y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
    n_classes = y_true_bin.shape[1]
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot ROC curves
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC Curves')
    plt.legend(loc="lower right", fontsize='small')
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def evaluate_model_comprehensive(model, test_loader, device, class_names, output_dir):
    """Comprehensive model evaluation."""
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Evaluating"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            
            _, preds = outputs.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_probs = np.array(all_probs)
    
    # Calculate overall metrics
    accuracy = 100. * sum(np.array(all_preds) == np.array(all_targets)) / len(all_targets)
    
    # Per-class metrics
    metrics_df = calculate_metrics_per_class(all_targets, all_preds, class_names)
    
    # Confusion matrix
    cm = plot_confusion_matrix(all_targets, all_preds, class_names, 
                              os.path.join(output_dir, 'confusion_matrix.png'))
    
    # ROC curves (for multiclass)
    try:
        plot_roc_curves_multiclass(all_targets, all_probs, class_names,
                                  os.path.join(output_dir, 'roc_curves.png'))
    except:
        print("Could not generate ROC curves (might be too many classes)")
    
    # Classification report
    report = classification_report(all_targets, all_preds, target_names=class_names, output_dict=True)
    
    # Save detailed metrics
    evaluation_results = {
        'overall_accuracy': accuracy,
        'per_class_metrics': metrics_df.to_dict(),
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'num_samples': len(all_targets),
        'num_correct': int(sum(np.array(all_preds) == np.array(all_targets))),
        'class_names': class_names
    }
    
    with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    # Print summary
    print(f"Overall Accuracy: {accuracy:.2f}%")
    print("\nPer-Class Metrics:")
    print(metrics_df.to_string(index=False))
    
    return evaluation_results


def analyze_prediction_confidence(model, test_loader, device, class_names):
    """Analyzes prediction confidence distribution."""
    model.eval()
    confidences = []
    correct_predictions = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Analyzing confidence"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            
            # Get max probability (confidence) and predicted class
            max_probs, preds = probs.max(1)
            
            confidences.extend(max_probs.cpu().numpy())
            correct_predictions.extend((preds == targets).cpu().numpy())
    
    # Plot confidence distribution
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(confidences, bins=50, alpha=0.7, edgecolor='black')
    plt.title('Distribution of Prediction Confidences')
    plt.xlabel('Confidence')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    correct_conf = [conf for i, conf in enumerate(confidences) if correct_predictions[i]]
    incorrect_conf = [conf for i, conf in enumerate(confidences) if not correct_predictions[i]]
    
    plt.hist(correct_conf, bins=30, alpha=0.7, label='Correct', color='green')
    plt.hist(incorrect_conf, bins=30, alpha=0.7, label='Incorrect', color='red')
    plt.title('Confidence by Correctness')
    plt.xlabel('Confidence')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confidence_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    avg_confidence = np.mean(confidences)
    avg_confidence_correct = np.mean(correct_conf) if correct_conf else 0
    avg_confidence_incorrect = np.mean(incorrect_conf) if incorrect_conf else 0
    
    print(f"\nConfidence Analysis:")
    print(f"Average confidence: {avg_confidence:.3f}")
    print(f"Average confidence (correct): {avg_confidence_correct:.3f}")
    print(f"Average confidence (incorrect): {avg_confidence_incorrect:.3f}")


def detect_overfitting(train_acc, val_acc):
    """Detects signs of overfitting."""
    gap = train_acc - val_acc
    
    if gap > 10:
        print("\n⚠️  WARNING: Potential overfitting detected!")
        print(f"Training accuracy: {train_acc:.2f}%")
        print(f"Validation accuracy: {val_acc:.2f}%")
        print(f"Difference: {gap:.2f}%")
        print("Consider adding regularization or data augmentation.")
    else:
        print(f"\n✅ No significant overfitting detected (gap: {gap:.2f}%)")
    
    return gap > 10


def generate_model_insights(model, test_loader, device, class_names):
    """Generates insights about model performance."""
    model.eval()
    
    # Track misclassifications
    misclassifications = []
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Analyzing misclassifications"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)
            
            for i in range(len(targets)):
                if preds[i] != targets[i]:
                    misclassifications.append({
                        'true_label': class_names[targets[i]],
                        'predicted_label': class_names[preds[i]],
                        'confidence': probs[i][preds[i]].item()
                    })
    
    # Count misclassification patterns
    from collections import Counter
    true_vs_pred = [(mc['true_label'], mc['predicted_label']) for mc in misclassifications]
    misclass_counts = Counter(true_vs_pred)
    
    print(f"\nTop 10 Misclassification Patterns:")
    for (true_label, pred_label), count in misclass_counts.most_common(10):
        print(f"  {true_label} → {pred_label}: {count} times")
    
    # Identify most confused classes
    confused_pairs = {}
    for (true_label, pred_label), count in misclass_counts.items():
        if true_label not in confused_pairs:
            confused_pairs[true_label] = {}
        confused_pairs[true_label][pred_label] = count
    
    print(f"\nMost Frequently Misidentified Classes:")
    for true_label, pred_counts in confused_pairs.items():
        total_misclassified = sum(pred_counts.values())
        if total_misclassified > 0:
            most_common_pred = max(pred_counts, key=pred_counts.get)
            print(f"  {true_label}: {total_misclassified} times (often as {most_common_pred})")


def main():
    parser = argparse.ArgumentParser(description="Evaluate cattle breed recognition model")
    parser.add_argument("--model-path", type=str, required=True, 
                       help="Path to trained model checkpoint")
    parser.add_argument("--test-dir", type=str, required=True, 
                       help="Path to test dataset directory")
    parser.add_argument("--model-name", type=str, default="resnet18",
                       choices=["resnet18", "mobilenet_v2", "efficientnet_b0"],
                       help="Model architecture")
    parser.add_argument("--batch-size", type=int, default=32, 
                       help="Batch size for evaluation")
    parser.add_argument("--output-dir", type=str, default="./evaluation_results",
                       help="Directory to save evaluation results")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    # First, we need to determine the number of classes from the checkpoint
    temp_checkpoint = torch.load(args.model_path, map_location=device)
    if 'class_names' in temp_checkpoint:
        num_classes = len(temp_checkpoint['class_names'])
        class_names = temp_checkpoint['class_names']
    else:
        # If class names are not in checkpoint, we need to infer from test directory
        test_dataset = CattleBreedDataset(Path(args.test_dir))
        num_classes = len(test_dataset.classes)
        class_names = test_dataset.classes
    
    model, checkpoint = load_trained_model(args.model_path, num_classes, args.model_name)
    model = model.to(device)
    
    print(f"Model loaded with {num_classes} classes")
    print(f"Classes: {class_names}")
    
    # Create test loader
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = CattleBreedDataset(Path(args.test_dir), transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Perform comprehensive evaluation
    print("Performing comprehensive evaluation...")
    results = evaluate_model_comprehensive(model, test_loader, device, class_names, args.output_dir)
    
    # Analyze prediction confidence
    print("Analyzing prediction confidence...")
    analyze_prediction_confidence(model, test_loader, device, class_names)
    
    # Generate model insights
    print("Generating model insights...")
    generate_model_insights(model, test_loader, device, class_names)
    
    print(f"\nEvaluation completed! Results saved to: {args.output_dir}")
    print(f"Overall Accuracy: {results['overall_accuracy']:.2f}%")


if __name__ == "__main__":
    main()