"""
beat_dataset.py
Beat extraction, resampling, and subject-level train/val/test splitting
for MIT-BIH and INCART arrhythmia databases.

Key principles:
- Subject-level splitting (no patient appears in more than one split)
- Beat windows extracted around R-peak annotations
- Single-lead (MLII for MIT-BIH, Lead II for INCART)
- Common sampling rate (250 Hz)
- Common beat window (0.8s before + 0.8s after = 1.6s)
"""
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter, defaultdict
import wfdb

from aami_mapping import map_symbol, AAMI_CLASSES

# ============================================================
# Configuration
# ============================================================
TARGET_FS = 250           # Common sampling frequency (Hz)
BEFORE_R = 0.8            # Seconds before R-peak
AFTER_R = 0.8             # Seconds after R-peak
BEAT_LEN = int((BEFORE_R + AFTER_R) * TARGET_FS)  # = 400 samples
RANDOM_SEED = 42

# Dataset paths
MITBIH_DIR = 'd:/git/mamintoosi-cs/holter-ecg-analysis/data/mit-bih'
INCART_DIR = 'd:/git/mamintoosi-cs/holter-ecg-analysis/data/incartdb'


def resample_signal(signal, orig_fs, target_fs):
    """Resample a 1D signal to target_fs using linear interpolation."""
    if orig_fs == target_fs:
        return signal
    duration = len(signal) / orig_fs
    n_samples = int(duration * target_fs)
    x_orig = np.linspace(0, duration, len(signal), endpoint=False)
    x_new = np.linspace(0, duration, n_samples, endpoint=False)
    return np.interp(x_new, x_orig, signal)


def extract_beat_window(signal, r_peak, orig_fs, target_fs, before_s, after_s):
    """Extract a beat window centered on r_peak.
    Returns None if window extends beyond signal boundaries."""
    before_samples = int(before_s * orig_fs)
    after_samples = int(after_s * orig_fs)
    start = r_peak - before_samples
    end = r_peak + after_samples
    
    if start < 0 or end >= len(signal):
        return None
    
    beat = signal[start:end].copy()
    # Resample to target
    beat_resampled = resample_signal(beat, orig_fs, target_fs)
    return beat_resampled


# ============================================================
# MIT-BIH Subject Mapping
# ============================================================
# Known MIT-BIH subject assignments from database documentation.
# Records 201+202, 203+215, 231+232, 233+234 share patients.

def get_mitbih_subject_map():
    """Return dict: subject_id -> list of record names."""
    subject_map = {}
    for rec in os.listdir(MITBIH_DIR):
        if not rec.endswith('.hea'):
            continue
        recname = rec.replace('.hea', '')
        num = int(recname)
        
        # Known same-patient groups
        if num in (201, 202):
            subj = 'S_201_202'
        elif num in (203, 215):
            subj = 'S_203_215'
        elif num in (231, 232):
            subj = 'S_231_232'
        elif num in (233, 234):
            subj = 'S_233_234'
        else:
            subj = f'S_{num}'
        
        if subj not in subject_map:
            subject_map[subj] = []
        subject_map[subj].append(recname)
    
    return subject_map


def get_incart_patient_map():
    """Return dict: patient_id -> list of record names."""
    patient_map = defaultdict(list)
    for rec in os.listdir(INCART_DIR):
        if not rec.endswith('.hea'):
            continue
        recname = rec.replace('.hea', '')
        try:
            r = wfdb.rdrecord(os.path.join(INCART_DIR, recname))
            patient_id = None
            for c in (r.comments or []):
                if 'patient' in c.lower():
                    parts = c.split()
                    for i, p in enumerate(parts):
                        if p.lower() == 'patient' and i + 1 < len(parts):
                            patient_id = parts[i + 1]
            if patient_id is None:
                patient_id = recname
            patient_map[patient_id].append(recname)
        except Exception:
            patient_map[recname].append(recname)
    
    return dict(patient_map)


# ============================================================
# Subject-Level Split
# ============================================================
def subject_level_split(subject_map, train_frac=0.70, val_frac=0.15, seed=RANDOM_SEED):
    """Split subjects into train/val/test at the subject level.
    Returns (train_subjects, val_subjects, test_subjects) as sets."""
    rng = np.random.RandomState(seed)
    subjects = list(subject_map.keys())
    rng.shuffle(subjects)
    
    n = len(subjects)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    
    train_subjs = set(subjects[:n_train])
    val_subjs = set(subjects[n_train:n_train + n_val])
    test_subjs = set(subjects[n_train + n_val:])
    
    return train_subjs, val_subjs, test_subjs


# ============================================================
# Beat Extraction
# ============================================================
def find_lead_index(sig_names, preferred_leads):
    """Find the index of the preferred lead in the signal list."""
    for preferred in preferred_leads:
        for i, name in enumerate(sig_names):
            if name.upper() == preferred.upper():
                return i
    return None


def extract_beats_for_record(record_name, data_dir, preferred_leads, beat_info_list):
    """Extract all valid beat windows from a single record.
    Appends to beat_info_list: (signal_window, aami_class, record, sample_idx, orig_sample)
    Returns count of beats extracted."""
    try:
        r = wfdb.rdrecord(os.path.join(data_dir, record_name))
        ann = wfdb.rdann(os.path.join(data_dir, record_name), 'atr')
    except Exception as e:
        print(f"  Warning: Could not read {record_name}: {e}")
        return 0
    
    # Find the right lead
    lead_idx = find_lead_index(r.sig_name, preferred_leads)
    if lead_idx is None:
        print(f"  Warning: Preferred leads {preferred_leads} not found in {record_name} (has {r.sig_name}). Skipping.")
        return 0
    
    signal = r.p_signal[:, lead_idx]
    orig_fs = r.fs
    
    n_extracted = 0
    for i, (sample_idx, sym) in enumerate(zip(ann.sample, ann.symbol)):
        aami_class = map_symbol(sym)
        if aami_class is None:
            continue  # Skip non-beat annotations
        
        beat = extract_beat_window(signal, int(sample_idx), orig_fs, TARGET_FS, BEFORE_R, AFTER_R)
        if beat is None:
            continue  # Skip beats near boundaries
        
        # Z-normalize the beat
        std = np.std(beat)
        if std < 1e-8:
            continue  # Skip flat beats
        beat = (beat - np.mean(beat)) / std
        
        beat_info_list.append({
            'signal': beat,
            'label': aami_class,
            'record': record_name,
            'sample_idx': i,
            'orig_sample': int(sample_idx),
        })
        n_extracted += 1
    
    return n_extracted


def extract_all_beats(subject_map, subjects, data_dir, preferred_leads, split_name=""):
    """Extract beats for all records belonging to given subjects.
    Returns list of beat info dicts and class counts."""
    beat_info = []
    records_used = []
    for subj in sorted(subjects):
        for rec in subject_map[subj]:
            n = extract_beats_for_record(rec, data_dir, preferred_leads, beat_info)
            if n > 0:
                records_used.append(rec)
    
    class_counts = Counter(b['label'] for b in beat_info)
    print(f"  {split_name}: {len(beat_info)} beats from {len(records_used)} records")
    for cls in AAMI_CLASSES:
        print(f"    {cls}: {class_counts.get(cls, 0)}")
    
    return beat_info, records_used, class_counts


# ============================================================
# Leakage Audit
# ============================================================
def leakage_audit(train_records, val_records, test_records, train_beats, val_beats, test_beats):
    """Verify no data leakage between splits."""
    print("\n" + "=" * 60)
    print("LEAKAGE AUDIT")
    print("=" * 60)
    
    train_rec_set = set(train_records)
    val_rec_set = set(val_records)
    test_rec_set = set(test_records)
    
    # Record-level checks
    train_val_overlap = train_rec_set & val_rec_set
    train_test_overlap = train_rec_set & test_rec_set
    val_test_overlap = val_rec_set & test_rec_set
    
    print(f"\nRecord overlaps:")
    print(f"  Train  intersect  Val:   {len(train_val_overlap)} records - {'PASS' if len(train_val_overlap)==0 else 'FAIL: '+str(train_val_overlap)}")
    print(f"  Train  intersect  Test:  {len(train_test_overlap)} records - {'PASS' if len(train_test_overlap)==0 else 'FAIL: '+str(train_test_overlap)}")
    print(f"  Val  intersect  Test:    {len(val_test_overlap)} records - {'PASS' if len(val_test_overlap)==0 else 'FAIL: '+str(val_test_overlap)}")
    
    all_ok = all(len(x) == 0 for x in [train_val_overlap, train_test_overlap, val_test_overlap])
    
    # Beat-level check: no beat should appear in multiple splits
    train_beat_keys = set((b['record'], b['orig_sample']) for b in train_beats)
    val_beat_keys = set((b['record'], b['orig_sample']) for b in val_beats)
    test_beat_keys = set((b['record'], b['orig_sample']) for b in test_beats)
    
    beat_tv = train_beat_keys & val_beat_keys
    beat_tt = train_beat_keys & test_beat_keys
    beat_vt = val_beat_keys & test_beat_keys
    
    print(f"\nBeat-level overlaps:")
    print(f"  Train  intersect  Val:   {len(beat_tv)} beats - {'PASS' if len(beat_tv)==0 else 'FAIL'}")
    print(f"  Train  intersect  Test:  {len(beat_tt)} beats - {'PASS' if len(beat_tt)==0 else 'FAIL'}")
    print(f"  Val  intersect  Test:    {len(beat_vt)} beats - {'PASS' if len(beat_vt)==0 else 'FAIL'}")
    
    all_ok = all_ok and all(len(x) == 0 for x in [beat_tv, beat_tt, beat_vt])
    
    if all_ok:
        print("\n[OK] NO DATA LEAKAGE DETECTED")
    else:
        print("\n[FAIL] DATA LEAKAGE DETECTED!")
    
    return all_ok


# ============================================================
# PyTorch Dataset
# ============================================================
class BeatDataset(Dataset):
    """PyTorch Dataset for single-lead beat windows."""
    
    def __init__(self, beat_info_list):
        """
        Args:
            beat_info_list: list of dicts with 'signal', 'label', 'record', etc.
        """
        self.beats = beat_info_list
        self.label_to_idx = {cls: i for i, cls in enumerate(AAMI_CLASSES)}
    
    def __len__(self):
        return len(self.beats)
    
    def __getitem__(self, idx):
        info = self.beats[idx]
        x = torch.tensor(info['signal'], dtype=torch.float32).unsqueeze(0)  # (1, beat_len)
        y = self.label_to_idx[info['label']]
        return x, y


def compute_class_weights(beat_info_list):
    """Compute class weights for weighted loss (from training data only).
    Inverse frequency weighting."""
    counts = Counter(b['label'] for b in beat_info_list)
    total = sum(counts.values())
    weights = []
    for cls in AAMI_CLASSES:
        c = counts.get(cls, 0)
        if c > 0:
            weights.append(total / c)
        else:
            weights.append(0.0)
    # Normalize so that average weight = 1.0
    avg = np.mean([w for w in weights if w > 0])
    weights = [w / avg if w > 0 else 0.0 for w in weights]
    return torch.tensor(weights, dtype=torch.float32)


# ============================================================
# Full Data Preparation Pipeline
# ============================================================
def prepare_mitbih_data(seed=RANDOM_SEED):
    """Prepare MIT-BIH data with subject-level splits.
    Returns (train_dataset, val_dataset, test_dataset, class_weights, metadata)."""
    print("\n" + "=" * 60)
    print("PREPARING MIT-BIH DATA")
    print("=" * 60)
    
    # 1. Get subject mapping
    subject_map = get_mitbih_subject_map()
    print(f"\nTotal subjects: {len(subject_map)}")
    
    # 2. Subject-level split
    train_subjs, val_subjs, test_subjs = subject_level_split(subject_map, seed=seed)
    
    print(f"\nTrain subjects ({len(train_subjs)}): {sorted(train_subjs)}")
    print(f"Val subjects ({len(val_subjs)}): {sorted(val_subjs)}")
    print(f"Test subjects ({len(test_subjs)}): {sorted(test_subjs)}")
    
    # 3. Verify subject disjointness
    assert train_subjs.isdisjoint(val_subjs), "Train-Val subject overlap!"
    assert train_subjs.isdisjoint(test_subjs), "Train-Test subject overlap!"
    assert val_subjs.isdisjoint(test_subjs), "Val-Test subject overlap!"
    print("\n[OK] Subject disjointness verified")
    
    # Preferred leads for MIT-BIH (MLII preferred, then V5/V1)
    preferred_leads = ['MLII', 'MLIIO', 'II', 'V5', 'V1']
    
    # 4. Extract beats for each split
    print("\n--- Extracting beats ---")
    train_beats, train_records, train_counts = extract_all_beats(
        subject_map, train_subjs, MITBIH_DIR, preferred_leads, "Train")
    val_beats, val_records, val_counts = extract_all_beats(
        subject_map, val_subjs, MITBIH_DIR, preferred_leads, "Val")
    test_beats, test_records, test_counts = extract_all_beats(
        subject_map, test_subjs, MITBIH_DIR, preferred_leads, "Test")
    
    # 5. Leakage audit
    leakage_audit(train_records, val_records, test_records, train_beats, val_beats, test_beats)
    
    # 6. Compute class weights from training data only
    class_weights = compute_class_weights(train_beats)
    print(f"\nClass weights (from training data):")
    for i, cls in enumerate(AAMI_CLASSES):
        print(f"  {cls}: {class_weights[i]:.4f}")
    
    # 7. Create datasets
    train_dataset = BeatDataset(train_beats)
    val_dataset = BeatDataset(val_beats)
    test_dataset = BeatDataset(test_beats)
    
    # 8. Metadata
    metadata = {
        'train_subjects': sorted(train_subjs),
        'val_subjects': sorted(val_subjs),
        'test_subjects': sorted(test_subjs),
        'train_records': sorted(train_records),
        'val_records': sorted(val_records),
        'test_records': sorted(test_records),
        'train_counts': dict(train_counts),
        'val_counts': dict(val_counts),
        'test_counts': dict(test_counts),
        'target_fs': TARGET_FS,
        'beat_len': BEAT_LEN,
        'before_r': BEFORE_R,
        'after_r': AFTER_R,
    }
    
    return train_dataset, val_dataset, test_dataset, class_weights, metadata


def prepare_incart_data(test_records_list=None):
    """Prepare INCART data as external test set.
    Patient-level grouping preserved.
    Returns (dataset, metadata)."""
    print("\n" + "=" * 60)
    print("PREPARING INCART EXTERNAL TEST DATA")
    print("=" * 60)
    
    patient_map = get_incart_patient_map()
    print(f"\nTotal patients: {len(patient_map)}")
    print(f"Total records: {sum(len(v) for v in patient_map.values())}")
    
    # Use Lead II for INCART (Lead II is MLII-equivalent)
    preferred_leads = ['II', 'MLII']
    
    # Extract beats for all INCART records
    all_beats = []
    all_records = []
    for patient_id in sorted(patient_map.keys(), key=lambda x: int(x) if x.isdigit() else 0):
        for rec in patient_map[patient_id]:
            n = extract_beats_for_record(rec, INCART_DIR, preferred_leads, all_beats)
            if n > 0:
                all_records.append(rec)
    
    class_counts = Counter(b['label'] for b in all_beats)
    print(f"\nINCART total: {len(all_beats)} beats from {len(all_records)} records")
    for cls in AAMI_CLASSES:
        print(f"  {cls}: {class_counts.get(cls, 0)}")
    
    dataset = BeatDataset(all_beats)
    
    metadata = {
        'patients': sorted(patient_map.keys()),
        'records': sorted(all_records),
        'patient_map': patient_map,
        'class_counts': dict(class_counts),
    }
    
    return dataset, metadata


if __name__ == '__main__':
    # Quick test: prepare data and verify
    train_ds, val_ds, test_ds, weights, meta = prepare_mitbih_data()
    
    print(f"\n--- Dataset sizes ---")
    print(f"Train: {len(train_ds)} beats")
    print(f"Val:   {len(val_ds)} beats")
    print(f"Test:  {len(test_ds)} beats")
    
    # Test a batch
    loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    x, y = next(iter(loader))
    print(f"\nBatch shape: {x.shape}, labels: {y.tolist()}")
    print(f"Label names: {[AAMI_CLASSES[i] for i in y.tolist()]}")
    
    # Save metadata
    os.makedirs('outputs/beat_classifier', exist_ok=True)
    with open('outputs/beat_classifier/mitbih_metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"\nMetadata saved to outputs/beat_classifier/mitbih_metadata.json")
    
    # Also prepare INCART
    incart_ds, incart_meta = prepare_incart_data()
    print(f"\nINCART dataset: {len(incart_ds)} beats")
    with open('outputs/beat_classifier/incart_metadata.json', 'w') as f:
        json.dump(incart_meta, f, indent=2)
    print("INCART metadata saved.")
