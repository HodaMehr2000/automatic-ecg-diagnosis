"""
Training pipeline for ECG multi-label classification.

Features:
    - ResNet1D with 6 output logits
    - BCEWithLogitsLoss with pos_weight for class imbalance
    - AdamW optimizer with ReduceLROnPlateau scheduler
    - Validation after each epoch
    - Best-model checkpointing based on validation loss
    - Reproducible seeds
    - CPU and CUDA support
    - Smoke test mode
"""

import os
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from resnet import build_ecg_resnet
from dataset_classifier import (
    load_part17_data,
    get_dataloaders,
    LABELS,
)


def set_seed(seed):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    n_samples = 0

    pbar = tqdm(loader, desc="Train", leave=False)
    for batch in pbar:
        x, y = batch[0].to(device), batch[1].to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        n_samples += bs

        pbar.set_postfix(loss=f"{total_loss / n_samples:.4f}")

    return total_loss / n_samples


def validate(model, loader, criterion, device):
    """Validate. Returns average loss."""
    model.eval()
    total_loss = 0.0
    n_samples = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Val  ", leave=False):
            x, y = batch[0].to(device), batch[1].to(device)
            logits = model(x)
            loss = criterion(logits, y)
            bs = x.size(0)
            total_loss += loss.item() * bs
            n_samples += bs

    return total_loss / n_samples


def train(args):
    """Main training loop."""
    set_seed(args.seed)

    # Device
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    print("\n--- Loading data ---")
    data = load_part17_data(args.csv_path, args.hdf5_path)
    train_loader, val_loader, test_loader, pos_weight, train_idx, val_idx, test_idx = (
        get_dataloaders(data, batch_size=args.batch_size, seed=args.seed)
    )

    print(f"\nTraining batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")

    # Model
    print("\n--- Building model ---")
    model = build_ecg_resnet(
        n_classes=len(LABELS),
        kernel_size=args.kernel_size,
        dropout_rate=args.dropout_rate,
        device=device,
    )
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss with class imbalance handling
    pos_weight = pos_weight.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"Using BCEWithLogitsLoss with pos_weight")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.patience,
        min_lr=args.min_lr,
        verbose=True,
    )

    # Output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save config
    config = {
        "n_classes": len(LABELS),
        "labels": LABELS,
        "kernel_size": args.kernel_size,
        "dropout_rate": args.dropout_rate,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "lr_factor": args.lr_factor,
        "patience": args.patience,
        "min_lr": args.min_lr,
        "epochs": args.epochs,
        "seed": args.seed,
        "pos_weight": pos_weight.cpu().tolist(),
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
    }
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Training loop
    print(f"\n--- Training for up to {args.epochs} epochs ---")
    best_val_loss = float("inf")
    best_epoch = -1
    history = []

    for epoch in range(args.epochs):
        start_time = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        elapsed = time.time() - start_time
        current_lr = optimizer.param_groups[0]["lr"]

        # Scheduler step
        scheduler.step(val_loss)

        # Log
        entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": current_lr,
            "time": elapsed,
        }
        history.append(entry)

        print(
            f"Epoch {epoch:3d}/{args.epochs - 1} | "
            f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
            f"LR: {current_lr:.2e} | Time: {elapsed:.1f}s"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "pos_weight": pos_weight.cpu().tolist(),
                },
                os.path.join(args.output_dir, "best_model.pth"),
            )
            print(f"  -> New best model saved (val_loss={val_loss:.6f})")

        # Save last model
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_loss": val_loss,
            },
            os.path.join(args.output_dir, "last_model.pth"),
        )

        # Save history
        import pandas as pd
        pd.DataFrame(history).to_csv(
            os.path.join(args.output_dir, "history.csv"), index=False
        )

        # Early stopping: if lr dropped below min_lr, stop
        if current_lr < args.min_lr:
            print(f"Learning rate {current_lr:.2e} below minimum {args.min_lr:.2e}. Stopping.")
            break

    print(f"\n--- Training complete ---")
    print(f"Best epoch: {best_epoch}, Best val_loss: {best_val_loss:.6f}")
    print(f"Best model saved to: {os.path.join(args.output_dir, 'best_model.pth')}")

    return model, pos_weight


def smoke_test(args):
    """Quick smoke test with small subset."""
    print("=== SMOKE TEST ===")
    set_seed(args.seed)

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load a small subset
    data = load_part17_data(args.csv_path, args.hdf5_path)

    # Use only first 100 samples for smoke test
    smoke_size = min(100, len(data["traces"]))
    data["traces"] = data["traces"][:smoke_size]
    data["labels"] = data["labels"][:smoke_size]
    data["exam_ids"] = data["exam_ids"][:smoke_size]
    data["patient_ids"] = data["patient_ids"][:smoke_size]

    train_loader, val_loader, _, pos_weight, _, _, _ = get_dataloaders(
        data, batch_size=8, seed=args.seed
    )

    model = build_ecg_resnet(n_classes=6, device=device)

    pos_weight = pos_weight.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Run 3 epochs
    print("\n--- Smoke test: 3 epochs ---")
    for epoch in range(3):
        model.train()
        total_loss = 0
        n = 0
        for batch in train_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
        avg_loss = total_loss / n

        # Validate
        model.eval()
        val_loss_total = 0
        vn = 0
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch[0].to(device), batch[1].to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss_total += loss.item() * x.size(0)
                vn += x.size(0)
        val_avg = val_loss_total / vn if vn > 0 else 0

        print(f"  Epoch {epoch}: train_loss={avg_loss:.4f}, val_loss={val_avg:.4f}")

    # Test checkpoint saving
    ckpt_path = os.path.join(args.output_dir, "smoke_test_ckpt.pth")
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save({"model": model.state_dict()}, ckpt_path)
    loaded = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(loaded["model"])
    os.remove(ckpt_path)

    # Test inference
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            x = batch[0].to(device)
            logits = model(x)
            probs = torch.sigmoid(logits)
            print(f"\n  Inference test:")
            print(f"    Logits shape: {logits.shape}")
            print(f"    Probs shape:  {probs.shape}")
            print(f"    Sample probs: {probs[0].cpu().numpy()}")
            break

    print("\n=== SMOKE TEST PASSED ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train ECG multi-label classifier."
    )
    parser.add_argument("--csv_path", default="data/exams.csv", help="Path to exams.csv")
    parser.add_argument("--hdf5_path", default="data/exams_part17.hdf5", help="Path to HDF5 file")
    parser.add_argument("--output_dir", default="outputs/classifier", help="Output directory")
    parser.add_argument("--epochs", type=int, default=70, help="Max epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--lr_factor", type=float, default=0.1, help="LR reduction factor")
    parser.add_argument("--patience", type=int, default=7, help="LR scheduler patience")
    parser.add_argument("--min_lr", type=float, default=1e-7, help="Minimum LR")
    parser.add_argument("--kernel_size", type=int, default=17, help="Conv kernel size")
    parser.add_argument("--dropout_rate", type=float, default=0.8, help="Dropout rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA")
    parser.add_argument("--smoke_test", action="store_true", help="Run smoke test only")
    args = parser.parse_args()

    if args.smoke_test:
        smoke_test(args)
    else:
        train(args)
