"""
train_beat_classifier.py
Training pipeline for single-lead ECG beat classification on MIT-BIH.
CPU-friendly: 5-10 epochs, batch_size=128, lightweight ResNet1D.

Usage:
    python train_beat_classifier.py
    python train_beat_classifier.py --epochs 10 --batch_size 64
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import Counter

from aami_mapping import AAMI_CLASSES
from beat_dataset import prepare_mitbih_data, BeatDataset, compute_class_weights
from resnet1d_beat import ResNet1D_Beat, count_parameters

# ============================================================
# Training Configuration
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description='Train beat classifier on MIT-BIH')
    parser.add_argument('--epochs', type=int, default=8, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', type=str, default='outputs/beat_classifier')
    parser.add_argument('--no_weighted_loss', action='store_true', help='Disable weighted loss')
    parser.add_argument('--smoke_test', action='store_true', help='Run 1-epoch smoke test')
    return parser.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Training Loop
# ============================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)
    
    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    """Evaluate on validation/test set. Returns (avg_loss, accuracy, all_preds, all_labels)."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    avg_loss = total_loss / total if total > 0 else float('inf')
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)


# ============================================================
# Main
# ============================================================
def main():
    args = parse_args()
    
    if args.smoke_test:
        args.epochs = 1
        print("[SMOKE TEST] Running 1 epoch only")
    
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    
    # 1. Prepare data
    train_ds, val_ds, test_ds, class_weights, metadata = prepare_mitbih_data(seed=args.seed)
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_ds)} beats")
    print(f"  Val:   {len(val_ds)} beats")
    print(f"  Test:  {len(test_ds)} beats")
    
    # 2. Create data loaders
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                            num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False,
                             num_workers=0, pin_memory=False)
    
    # 3. Create model
    model = ResNet1D_Beat(n_classes=len(AAMI_CLASSES), dropout=args.dropout)
    model = model.to(device)
    print(f"\nModel parameters: {count_parameters(model):,}")
    
    # 4. Loss function
    if args.no_weighted_loss:
        criterion = nn.CrossEntropyLoss()
        print("Loss: CrossEntropy (unweighted)")
    else:
        class_weights_dev = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_dev)
        print(f"Loss: CrossEntropy (weighted)")
        print(f"  Weights: {dict(zip(AAMI_CLASSES, class_weights.tolist()))}")
    
    # 5. Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # 6. Training loop
    os.makedirs(args.output_dir, exist_ok=True)
    best_val_loss = float('inf')
    best_epoch = -1
    patience_counter = 0
    history = []
    
    print(f"\n{'='*60}")
    print(f"TRAINING: {args.epochs} epochs, batch_size={args.batch_size}, lr={args.lr}")
    print(f"{'='*60}\n")
    
    start_total = time.time()
    
    for epoch in range(args.epochs):
        start_epoch = time.time()
        
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, val_preds, val_labels = evaluate(model, val_loader, criterion, device)
        
        elapsed = time.time() - start_epoch
        
        # Per-class validation accuracy
        val_per_class = {}
        for i, cls in enumerate(AAMI_CLASSES):
            mask = val_labels == i
            if mask.sum() > 0:
                val_per_class[cls] = (val_preds[mask] == i).mean()
            else:
                val_per_class[cls] = 0.0
        
        # Log
        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
              f"Time: {elapsed:.1f}s")
        
        # Print per-class val accuracy
        per_class_str = "  Per-class Val Acc: " + " | ".join(
            f"{cls}:{val_per_class[cls]:.3f}" for cls in AAMI_CLASSES if val_per_class[cls] > 0
        )
        print(per_class_str)
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_per_class': val_per_class,
            'time': elapsed,
        })
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Early stopping / model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'class_weights': class_weights,
                'args': vars(args),
                'metadata': metadata,
            }, os.path.join(args.output_dir, 'best_model.pth'))
            print(f"  [OK] Saved best model (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n  [STOP] Early stopping at epoch {epoch+1} (no improvement for {args.patience} epochs)")
                break
        
        print()
    
    total_time = time.time() - start_total
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Best epoch: {best_epoch} (val_loss={best_val_loss:.4f})")
    print(f"{'='*60}")
    
    # 7. Final evaluation on test set (using best model)
    print(f"\n--- Loading best model and evaluating on TEST set ---")
    checkpoint = torch.load(os.path.join(args.output_dir, 'best_model.pth'), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    
    # 8. Save training history and results
    results = {
        'config': vars(args),
        'history': history,
        'best_epoch': best_epoch,
        'test_loss': test_loss,
        'test_acc': test_acc,
        'test_predictions': test_preds.tolist(),
        'test_labels': test_labels.tolist(),
        'metadata': metadata,
    }
    
    with open(os.path.join(args.output_dir, 'training_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Also save config
    config = {
        'model': 'ResNet1D_Beat',
        'n_classes': len(AAMI_CLASSES),
        'classes': AAMI_CLASSES,
        'input_channels': 1,
        'input_length': metadata['beat_len'],
        'target_fs': metadata['target_fs'],
        'model_params': count_parameters(model),
        'best_epoch': best_epoch,
        'test_acc': test_acc,
        'test_loss': test_loss,
        'class_weights': class_weights.tolist(),
    }
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nResults saved to {args.output_dir}/")
    print(f"  training_results.json")
    print(f"  config.json")
    print(f"  best_model.pth")
    
    return test_acc


if __name__ == '__main__':
    main()
