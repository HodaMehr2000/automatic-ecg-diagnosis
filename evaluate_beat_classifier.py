"""
evaluate_beat_classifier.py
Evaluation of the beat classifier on MIT-BIH held-out test set.
Reports: confusion matrix, per-class precision/recall/F1, macro F1, balanced accuracy.
"""
import os
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from collections import Counter

from aami_mapping import AAMI_CLASSES
from beat_dataset import prepare_mitbih_data
from resnet1d_beat import ResNet1D_Beat


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate beat classifier on MIT-BIH test set')
    parser.add_argument('--model_path', type=str, default='outputs/beat_classifier/best_model.pth')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--output_dir', type=str, default='outputs/beat_classifier')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def compute_metrics(all_labels, all_preds, class_names):
    """Compute per-class and overall metrics."""
    n_classes = len(class_names)
    
    # Confusion matrix
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(all_labels, all_preds):
        cm[true, pred] += 1
    
    # Per-class metrics
    results = {}
    for i, cls in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # sensitivity
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        support = cm[i, :].sum()
        
        results[cls] = {
            'precision': precision,
            'recall': recall,
            'sensitivity': recall,
            'specificity': specificity,
            'f1': f1,
            'support': int(support),
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'tn': int(tn),
        }
    
    # Macro averages
    support_mask = np.array([results[cls]['support'] > 0 for cls in class_names])
    active_classes = [cls for cls in class_names if results[cls]['support'] > 0]
    
    macro_precision = np.mean([results[cls]['precision'] for cls in active_classes]) if active_classes else 0
    macro_recall = np.mean([results[cls]['recall'] for cls in active_classes]) if active_classes else 0
    macro_f1 = np.mean([results[cls]['f1'] for cls in active_classes]) if active_classes else 0
    balanced_acc = np.mean([results[cls]['recall'] for cls in active_classes]) if active_classes else 0
    
    # Weighted averages
    total = sum(results[cls]['support'] for cls in active_classes)
    weighted_precision = sum(results[cls]['precision'] * results[cls]['support'] for cls in active_classes) / total if total > 0 else 0
    weighted_f1 = sum(results[cls]['f1'] * results[cls]['support'] for cls in active_classes) / total if total > 0 else 0
    
    overall_accuracy = np.trace(cm) / cm.sum() if cm.sum() > 0 else 0
    
    summary = {
        'accuracy': overall_accuracy,
        'balanced_accuracy': balanced_acc,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_precision,
        'weighted_f1': weighted_f1,
        'total_samples': int(cm.sum()),
    }
    
    return cm, results, summary


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 1. Load model
    print(f"\nLoading model from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    
    # Determine model config from checkpoint
    saved_args = checkpoint.get('args', {})
    saved_metadata = checkpoint.get('metadata', {})
    dropout = saved_args.get('dropout', 0.3)
    
    model = ResNet1D_Beat(n_classes=len(AAMI_CLASSES), dropout=dropout)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"  Model loaded from epoch {checkpoint.get('epoch', '?')}")
    print(f"  Val loss: {checkpoint.get('val_loss', '?'):.4f}, Val acc: {checkpoint.get('val_acc', '?'):.4f}")
    
    # 2. Prepare test data
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    _, _, test_ds, _, metadata = prepare_mitbih_data(seed=args.seed)
    
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"\nTest set: {len(test_ds)} beats")
    
    # 3. Inference
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # 4. Compute metrics
    cm, per_class, summary = compute_metrics(all_labels, all_preds, AAMI_CLASSES)
    
    # 5. Print results
    print(f"\n{'='*70}")
    print("MIT-BIH TEST SET EVALUATION RESULTS")
    print(f"{'='*70}")
    
    print(f"\nOverall Accuracy:        {summary['accuracy']:.4f}")
    print(f"Balanced Accuracy:       {summary['balanced_accuracy']:.4f}")
    print(f"Macro F1:                {summary['macro_f1']:.4f}")
    print(f"Weighted F1:             {summary['weighted_f1']:.4f}")
    print(f"Total test samples:      {summary['total_samples']}")
    
    print(f"\n{'Class':>6} {'Prec':>8} {'Rec':>8} {'Spec':>8} {'F1':>8} {'Support':>8}")
    print("-" * 50)
    for cls in AAMI_CLASSES:
        m = per_class[cls]
        if m['support'] > 0:
            print(f"{cls:>6} {m['precision']:>8.4f} {m['recall']:>8.4f} {m['specificity']:>8.4f} {m['f1']:>8.4f} {m['support']:>8d}")
        else:
            print(f"{cls:>6} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8} {0:>8d}")
    
    # Confusion matrix
    print(f"\nConfusion Matrix (rows=true, cols=pred):")
    active = [cls for cls in AAMI_CLASSES if per_class[cls]['support'] > 0]
    header = f"{'':>6}" + "".join(f"{cls:>8}" for cls in active)
    print(header)
    for i, cls in enumerate(AAMI_CLASSES):
        if per_class[cls]['support'] == 0:
            continue
        row = f"{cls:>6}"
        for j, cls2 in enumerate(AAMI_CLASSES):
            if per_class[cls2]['support'] == 0:
                continue
            row += f"{cm[i, j]:>8d}"
        print(row)
    
    # 6. Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    results = {
        'summary': summary,
        'per_class': per_class,
        'confusion_matrix': cm.tolist(),
        'test_records': metadata.get('test_records', []),
        'test_subjects': metadata.get('test_subjects', []),
        'test_counts': metadata.get('test_counts', {}),
        'model_path': args.model_path,
    }
    
    with open(os.path.join(args.output_dir, 'mitbih_test_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {args.output_dir}/mitbih_test_results.json")
    
    return summary


if __name__ == '__main__':
    main()
