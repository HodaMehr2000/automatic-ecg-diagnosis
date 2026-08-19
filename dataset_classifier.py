"""
PyTorch Dataset for ECG multi-label classification.

Handles HDF5/CSV alignment, patient-level train/val/test splitting,
and provides DataLoaders for the classification pipeline.
"""

import os
import numpy as np
import pandas as pd
import h5py
import torch
from torch.utils.data import Dataset, DataLoader

# Target abnormalities
LABELS = ["1dAVb", "RBBB", "LBBB", "SB", "ST", "AF"]

# Default HDF5 dataset key for ECG traces
HDF5_TRACES_KEY = "tracings"
HDF5_EXAM_ID_KEY = "exam_id"


class ECGClassificationDataset(Dataset):
    """
    PyTorch Dataset for ECG multi-label classification.

    Loads ECG traces from HDF5 and aligns them with CSV labels using exam_id.
    Returns tensors in (channels, samples) format for Conv1D.
    """

    def __init__(self, traces, labels, exam_ids=None):
        """
        Args:
            traces: numpy array of shape (N, 4096, 12) or (N, 12, 4096).
            labels: numpy array of shape (N, 6) with binary labels.
            exam_ids: optional numpy array of exam IDs for traceability.
        """
        # Ensure shape is (N, 12, 4096) for PyTorch Conv1D: (batch, channels, samples)
        if traces.ndim == 3 and traces.shape[2] == 12:
            # HDF5 stores as (N, 4096, 12) -> transpose to (N, 12, 4096)
            traces = traces.transpose(0, 2, 1)

        self.traces = torch.tensor(traces, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.exam_ids = exam_ids

    def __len__(self):
        return len(self.traces)

    def __getitem__(self, idx):
        x = self.traces[idx]
        y = self.labels[idx]
        if self.exam_ids is not None:
            return x, y, self.exam_ids[idx]
        return x, y


def load_part17_data(csv_path, hdf5_path):
    """
    Load and align part17 data from CSV and HDF5 files.

    Uses exam_id for alignment (not positional).

    Args:
        csv_path: Path to exams.csv.
        hdf5_path: Path to exams_part17.hdf5.

    Returns:
        dict with keys:
            'traces': numpy array (N, 4096, 12)
            'labels': numpy array (N, 6)
            'exam_ids': numpy array (N,)
            'patient_ids': numpy array (N,)
            'df': filtered DataFrame
    """
    # Load CSV and filter to part17
    df = pd.read_csv(csv_path)
    df_part17 = df[df["trace_file"] == os.path.basename(hdf5_path)].copy()
    df_part17 = df_part17.reset_index(drop=True)

    print(f"Loaded {len(df_part17)} rows from CSV for {os.path.basename(hdf5_path)}")

    # Load HDF5
    f = h5py.File(hdf5_path, "r")
    h5_exam_ids = f[HDF5_EXAM_ID_KEY][:]
    h5_traces = f[HDF5_TRACES_KEY]

    # Build a mapping from exam_id -> HDF5 index
    h5_id_to_idx = {eid: idx for idx, eid in enumerate(h5_exam_ids)}

    # Align CSV rows with HDF5 traces using exam_id
    aligned_traces = []
    aligned_labels = []
    aligned_exam_ids = []
    aligned_patient_ids = []
    skipped = 0

    for _, row in df_part17.iterrows():
        exam_id = row["exam_id"]
        if exam_id in h5_id_to_idx:
            idx = h5_id_to_idx[exam_id]
            aligned_traces.append(h5_traces[idx])
            aligned_labels.append([int(row[label]) for label in LABELS])
            aligned_exam_ids.append(exam_id)
            aligned_patient_ids.append(row["patient_id"])
        else:
            skipped += 1

    f.close()

    if skipped > 0:
        print(f"Warning: {skipped} CSV rows had no matching HDF5 record")

    traces = np.array(aligned_traces, dtype=np.float32)
    labels = np.array(aligned_labels, dtype=np.float32)
    exam_ids = np.array(aligned_exam_ids)
    patient_ids = np.array(aligned_patient_ids)

    print(f"Aligned {len(traces)} ECG records with labels")
    print(f"Traces shape: {traces.shape}, Labels shape: {labels.shape}")

    return {
        "traces": traces,
        "labels": labels,
        "exam_ids": exam_ids,
        "patient_ids": patient_ids,
        "df": df_part17,
    }


def patient_level_split(patient_ids, train_ratio=0.70, val_ratio=0.15, seed=42):
    """
    Split data at the patient level to prevent data leakage.

    Each patient's ECGs go entirely into one split.

    Args:
        patient_ids: array of patient IDs for each sample.
        train_ratio: fraction for training.
        val_ratio: fraction for validation.
        seed: random seed for reproducibility.

    Returns:
        train_idx, val_idx, test_idx: arrays of sample indices.
    """
    rng = np.random.RandomState(seed)

    unique_patients = np.unique(patient_ids)
    n_patients = len(unique_patients)

    # Shuffle patients
    shuffled = rng.permutation(unique_patients)

    n_train = int(n_patients * train_ratio)
    n_val = int(n_patients * val_ratio)

    train_patients = set(shuffled[:n_train])
    val_patients = set(shuffled[n_train : n_train + n_val])
    test_patients = set(shuffled[n_train + n_val :])

    train_idx = np.array([i for i, p in enumerate(patient_ids) if p in train_patients])
    val_idx = np.array([i for i, p in enumerate(patient_ids) if p in val_patients])
    test_idx = np.array([i for i, p in enumerate(patient_ids) if p in test_patients])

    print(f"\nPatient-level split ({n_patients} unique patients):")
    print(f"  Train: {len(train_patients)} patients, {len(train_idx)} ECGs")
    print(f"  Val:   {len(val_patients)} patients, {len(val_idx)} ECGs")
    print(f"  Test:  {len(test_patients)} patients, {len(test_idx)} ECGs")

    return train_idx, val_idx, test_idx


def compute_pos_weight(labels):
    """
    Compute positive class weights for BCEWithLogitsLoss from training labels.

    pos_weight = (#negatives) / (#positives) per class.

    Args:
        labels: numpy array of shape (N, 6) with binary labels.

    Returns:
        torch tensor of shape (6,) with pos_weight values.
    """
    n_pos = labels.sum(axis=0)
    n_neg = labels.shape[0] - n_pos
    # Avoid division by zero: if no positives, set weight to 1
    pos_weight = np.ones_like(n_pos, dtype=np.float64)
    nonzero = n_pos > 0
    pos_weight[nonzero] = n_neg[nonzero] / n_pos[nonzero]
    print(f"\nClass distribution (training set):")
    for i, label in enumerate(LABELS):
        print(f"  {label}: {int(n_pos[i])} pos, {int(n_neg[i])} neg, "
              f"pos_weight={pos_weight[i]:.2f}")
    return torch.tensor(pos_weight, dtype=torch.float32)


def get_dataloaders(data, batch_size=32, num_workers=0, seed=42):
    """
    Create train/val/test DataLoaders with patient-level splitting.

    Args:
        data: dict from load_part17_data().
        batch_size: batch size.
        num_workers: number of data loading workers.
        seed: random seed.

    Returns:
        train_loader, val_loader, test_loader: DataLoader instances.
        pos_weight: tensor of positive class weights (computed from training set).
        train_idx, val_idx, test_idx: index arrays.
    """
    train_idx, val_idx, test_idx = patient_level_split(
        data["patient_ids"], seed=seed
    )

    # Create datasets
    train_dataset = ECGClassificationDataset(
        data["traces"][train_idx],
        data["labels"][train_idx],
        data["exam_ids"][train_idx],
    )
    val_dataset = ECGClassificationDataset(
        data["traces"][val_idx],
        data["labels"][val_idx],
        data["exam_ids"][val_idx],
    )
    test_dataset = ECGClassificationDataset(
        data["traces"][test_idx],
        data["labels"][test_idx],
        data["exam_ids"][test_idx],
    )

    # Compute pos_weight from training set only
    pos_weight = compute_pos_weight(data["labels"][train_idx])

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, pos_weight, train_idx, val_idx, test_idx


if __name__ == "__main__":
    # Quick test
    data_dir = "data"
    csv_path = os.path.join(data_dir, "exams.csv")
    hdf5_path = os.path.join(data_dir, "exams_part17.hdf5")

    data = load_part17_data(csv_path, hdf5_path)

    train_loader, val_loader, test_loader, pos_weight, _, _, _ = get_dataloaders(data, batch_size=4)

    # Test loading a batch
    batch = next(iter(train_loader))
    print(f"\nBatch shapes: x={batch[0].shape}, y={batch[1].shape}")
    print(f"Sample labels (first 5):")
    for i in range(min(5, len(batch[1]))):
        active = [LABELS[j] for j in range(len(LABELS)) if batch[1][i][j] > 0]
        print(f"  Sample {i}: {active if active else 'none'}")
