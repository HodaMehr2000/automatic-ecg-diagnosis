"""
Evaluation script for ECG multi-label classification.

Produces comprehensive metrics per class and a prediction CSV for inspection.
Metrics: AUROC, precision, recall/sensitivity, specificity, F1, positive count.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from resnet import build_ecg_resnet
from dataset_classifier import (
    load_part17_data,
    get_dataloaders,
    LABELS,
)


def compute_specificity(y_true, y_pred):
    """Compute specificity (true negative rate) for each class."""
    specificity = []
    for i in range(y_true.shape[1]):
        tn, fp, fn, tp = confusion_matrix(
            y_true[:, i], y_pred[:, i], labels=[0, 1]
        ).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificity.append(spec)
    return np.array(specificity)


def _compute_metrics(all_labels, all_probs, all_preds):
    """Compute per-class and macro metrics."""
    results = {}
    aurocs = []
    precisions = []
    recalls = []
    specificities = []
    f1s = []

    for i, label in enumerate(LABELS):
        y_true = all_labels[:, i]
        y_prob = all_probs[:, i]
        y_pred = all_preds[:, i]
        n_pos = int(y_true.sum())

        if n_pos > 0 and n_pos < len(y_true):
            auroc = roc_auc_score(y_true, y_prob)
        else:
            auroc = float("nan")

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        specificity = compute_specificity(y_true.reshape(-1, 1), y_pred.reshape(-1, 1))[0]

        aurocs.append(auroc)
        precisions.append(precision)
        recalls.append(recall)
        specificities.append(specificity)
        f1s.append(f1)

        results[label] = {
            "auroc": round(auroc, 4) if not np.isnan(auroc) else None,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "specificity": round(specificity, 4),
            "f1": round(f1, 4),
            "n_positives": n_pos,
        }

    valid_aurocs = [a for a in aurocs if not np.isnan(a)]
    results["macro"] = {
        "auroc": round(np.mean(valid_aurocs), 4) if valid_aurocs else None,
        "precision": round(np.mean(precisions), 4),
        "recall": round(np.mean(recalls), 4),
        "specificity": round(np.mean(specificities), 4),
        "f1": round(np.mean(f1s), 4),
    }
    return results


def evaluate(model, loader, device, threshold=0.5):
    """
    Evaluate model on a dataset.

    Args:
        model: trained model.
        loader: DataLoader.
        device: torch device.
        threshold: classification threshold.

    Returns:
        dict with per-class and macro metrics, plus raw predictions.
    """
    model.eval()
    all_logits = []
    all_labels = []
    all_exam_ids = []

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                x, y, exam_ids = batch
                all_exam_ids.extend(exam_ids.tolist())
            else:
                x, y = batch
            x = x.to(device)
            logits = model(x)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(y.numpy())

    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_probs = 1.0 / (1.0 + np.exp(-all_logits))  # sigmoid
    if threshold is not None:
        all_preds = (all_probs >= threshold).astype(int)
    else:
        all_preds = None

    # If no threshold, just return raw data (caller computes metrics)
    if all_preds is None:
        return {}, all_probs, all_labels, None, all_exam_ids

    # Per-class metrics
    results = _compute_metrics(all_labels, all_probs, all_preds)

    return results, all_probs, all_labels, all_preds, all_exam_ids


def optimize_thresholds(model, val_loader, device):
    """
    Optimize classification thresholds on validation set using F1.

    Returns:
        array of 6 optimized thresholds.
    """
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            logits = model(x)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(y.numpy())

    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_probs = 1.0 / (1.0 + np.exp(-all_logits))

    thresholds = np.full(len(LABELS), 0.5)
    for i in range(len(LABELS)):
        best_f1 = 0
        best_t = 0.5
        for t in np.arange(0.05, 0.95, 0.05):
            preds = (all_probs[:, i] >= t).astype(int)
            f1 = f1_score(all_labels[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        thresholds[i] = best_t
        print(f"  {LABELS[i]}: threshold={best_t:.2f} (best F1={best_f1:.4f})")

    return thresholds


def save_predictions(output_path, exam_ids, labels, probs, preds, patient_ids=None):
    """
    Save prediction CSV for inspection.
    """
    records = []
    for i in range(len(exam_ids)):
        row = {"exam_id": int(exam_ids[i])}
        if patient_ids is not None:
            row["patient_id"] = int(patient_ids[i]) if i < len(patient_ids) else ""
        for j, label in enumerate(LABELS):
            row[f"true_{label}"] = int(labels[i, j])
            row[f"prob_{label}"] = round(float(probs[i, j]), 6)
            row[f"pred_{label}"] = int(preds[i, j])
        records.append(row)

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"\nPredictions saved to: {output_path}")
    return df


def main(args):
    # Device
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    print("\n--- Loading data ---")
    data = load_part17_data(args.csv_path, args.hdf5_path)

    # Create test loader (we need the indices to get patient_ids for test set)
    train_loader, val_loader, test_loader, pos_weight, train_idx, val_idx, test_idx = (
        get_dataloaders(data, batch_size=args.batch_size, seed=args.seed)
    )

    test_patient_ids = data["patient_ids"][test_idx]
    test_exam_ids = data["exam_ids"][test_idx]

    # Load model
    print("\n--- Loading model ---")
    model = build_ecg_resnet(
        n_classes=len(LABELS),
        kernel_size=args.kernel_size,
        dropout_rate=args.dropout_rate,
        device=device,
    )

    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded model from: {args.model_path}")
    print(f"Checkpoint epoch: {ckpt.get('epoch', 'N/A')}, val_loss: {ckpt.get('val_loss', 'N/A')}")

    # Determine thresholds
    if args.optimize_thresholds:
        print("\n--- Optimizing thresholds on validation set ---")
        thresholds = optimize_thresholds(model, val_loader, device)
    else:
        thresholds = np.full(len(LABELS), 0.5)
        print(f"\nUsing default threshold: 0.5")

    print(f"Thresholds: {dict(zip(LABELS, thresholds.round(2).tolist()))}")

    # Evaluate on test set
    print("\n--- Evaluating on TEST set ---")
    _, probs, labels, _, eval_exam_ids = evaluate(
        model, test_loader, device, threshold=None
    )

    # Apply per-class thresholds and compute metrics
    preds = (probs >= thresholds).astype(int)
    results = _compute_metrics(labels, probs, preds)

    # Add threshold info to results
    for i, label in enumerate(LABELS):
        if label in results:
            results[label]["threshold"] = round(float(thresholds[i]), 2)

    # Print results
    print("\n" + "=" * 90)
    print(f"{'Class':<10} {'AUROC':>8} {'Prec':>8} {'Recall':>8} {'Spec':>8} {'F1':>8} {'#Pos':>6}")
    print("-" * 90)
    for label in LABELS:
        r = results[label]
        auroc_str = f"{r['auroc']:.4f}" if r["auroc"] is not None else "N/A"
        print(
            f"{label:<10} {auroc_str:>8} {r['precision']:>8.4f} {r['recall']:>8.4f} "
            f"{r['specificity']:>8.4f} {r['f1']:>8.4f} {r['n_positives']:>6d}"
        )
    print("-" * 90)
    m = results["macro"]
    auroc_str = f"{m['auroc']:.4f}" if m["auroc"] is not None else "N/A"
    print(
        f"{'MACRO':<10} {auroc_str:>8} {m['precision']:>8.4f} {m['recall']:>8.4f} "
        f"{m['specificity']:>8.4f} {m['f1']:>8.4f}"
    )
    print("=" * 90)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "test_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Save predictions CSV
    pred_path = os.path.join(args.output_dir, "predictions.csv")
    save_predictions(
        pred_path,
        test_exam_ids,
        labels,
        probs,
        preds,
        patient_ids=test_patient_ids,
    )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ECG multi-label classifier.")
    parser.add_argument("--model_path", default="outputs/classifier/best_model.pth",
                        help="Path to model checkpoint")
    parser.add_argument("--csv_path", default="data/exams.csv", help="Path to exams.csv")
    parser.add_argument("--hdf5_path", default="data/exams_part17.hdf5", help="Path to HDF5 file")
    parser.add_argument("--output_dir", default="outputs/classifier", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--kernel_size", type=int, default=17, help="Conv kernel size")
    parser.add_argument("--dropout_rate", type=float, default=0.8, help="Dropout rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA")
    parser.add_argument("--optimize_thresholds", action="store_true",
                        help="Optimize thresholds on validation set")
    args = parser.parse_args()

    main(args)
