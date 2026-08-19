"""
inspect_arrhythmia_datasets.py
Inspect MIT-BIH Arrhythmia Database and St. Petersburg INCART Arrhythmia Database.
Report records, annotations, subjects, leads, sampling rates.
"""
import os
import struct
import glob
import numpy as np
from collections import Counter, defaultdict

# ============================================================
# WFDB Header Parser (manual, no external dependency)
# ============================================================
def parse_hea_header(hea_path):
    """Parse a WFDB .hea header file."""
    with open(hea_path, 'r') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    
    # First line: record_name n_signals sampling_frequency n_samples
    parts = lines[0].split()
    record_name = parts[0]
    n_signals = int(parts[1])
    fs = int(float(parts[2]))
    n_samples = int(parts[3]) if len(parts) > 3 else None
    
    signals = []
    comments = []
    for line in lines[1:]:
        if line.startswith('#'):
            comments.append(line)
            continue
        if not line or line.startswith('#'):
            continue
        sparts = line.split()
        if len(sparts) < 10:
            continue
        sig_name = sparts[8]
        fmt = sparts[2]
        gain = float(sparts[3])
        baseline = int(sparts[4])
        sig_size = int(sparts[6])
        signals.append({
            'name': sig_name,
            'fmt': fmt,
            'gain': gain,
            'baseline': baseline,
            'size': sig_size,
        })
    
    return {
        'record_name': record_name,
        'n_signals': n_signals,
        'fs': fs,
        'n_samples': n_samples,
        'signals': signals,
        'comments': comments,
    }


def parse_atr_annotation(atr_path, n_samples):
    """
    Parse WFDB binary annotation file (.atr).
    Returns list of (sample_position, symbol).
    """
    with open(atr_path, 'rb') as f:
        data = f.read()
    
    annotations = []
    i = 0
    sample = 0
    
    while i < len(data):
        byte0 = data[i]
        
        # Check for leader byte 0
        if byte0 == 0:
            i += 1
            if i >= len(data):
                break
            byte0 = data[i]
        
        # Byte 0 encodes annotation type in bits 6-5 (or all bits)
        # In MIT format, the first byte is the annotation code
        # 0-31: standard annotation codes
        # Byte0 & 0x60 = annotation type, Byte0 & 0x1F = annotation code
        # Actually for MIT format:
        # Byte 0: bits[6:5] = type (0=A+, 1=V+, 2=VF+, 3=F+)
        #         bits[4:0] = annotation code (for aux data: 0x3F=string, 0x3E=lev, 0x39=param)
        
        annot_code = byte0 & 0x1F  # lower 5 bits
        annot_type = (byte0 >> 5) & 0x03  # bits 6-5
        
        if byte0 == 0x59:  # 0x59 = 'Y' = pacer spike
            i += 1
            # Skip additional byte
            if i < len(data):
                # Next two bytes are signed 16-bit interval
                if i + 1 < len(data):
                    interval = struct.unpack_from('<h', data, i)[0]
                    sample += interval
                    i += 2
                    annotations.append((sample, 'P'))
                    continue
            break
        
        if byte0 == 0x14:  # 0x14 = cue/alert
            i += 1
            if i < len(data):
                # skip extra byte
                aux_byte = data[i]
                if aux_byte != 0:
                    i += aux_byte
            continue
        
        # Standard annotation
        if i + 1 < len(data):
            interval = struct.unpack_from('<h', data, i)[1]
            sample += interval
            i += 2
        
        if i < len(data) and annot_code <= 59:
            # Byte 2 is the symbol byte for standard annotations
            symbol_byte = data[i]
            i += 1
            
            # Map annotation code to symbol
            sym = annot_code_to_symbol(annot_code, symbol_byte)
            annotations.append((sample, sym))
        else:
            break
    
    return annotations


def annot_code_to_symbol(code, symbol_byte):
    """Map annotation code to MIT-BIH symbol."""
    # MIT-BIH annotation symbols
    symbol_map = {
        0: 'N',    # Normal beat
        1: 'L',    # Left bundle branch block
        2: 'R',    # Right bundle branch block
        3: 'a',    # Atrial premature beat
        4: 'V',    # Premature ventricular contraction
        5: 'F',    # Fusion of ventricular and normal beat
        6: 'J',    # Nodal (junctional) premature beat
        7: 'S',    # Supraventricular premature beat
        8: '/',    # Paced beat
        9: 'Q',    # Unclassifiable beat
        10: '~',   # Signal quality change
        11: '!',   # Ventricular flutter wave
        12: '[',   # Start of ventricular flutter/fibrillation
        13: ']',   # End of ventricular flutter/fibrillation
        14: 'x',   # Non-conducted P-wave (not a beat)
        15: '(',   # Waveform onset
        16: ')',   # Waveform end
        17: 'p',   # P-wave peak
        18: 't',   # T-wave peak
        19: 'u',   # U-wave peak
        20: '`',   # MSYS peak (not a beat)
        21: '\'',  # MSYS peak (not a beat)
        22: '^',   # Non-conducted pacemaker spike (not a beat)
        23: ',',   # Rhythm change
        24: 'w',   # Wandering baseline
        25: 'i',   # Electrode disconnect
        26: 's',   # ST change
        27: 'T',   # T-wave change
        28: '*',   # Heart rate change (not a beat)
        29: 'D',   # Diastolic timing (not a beat)
        30: '"',   # Systolic timing (not a beat)
        31: '=',   # Measurement annotation (not a beat)
        59: 'P',   # Pacer spike (sometimes)
    }
    
    if code in symbol_map:
        return symbol_map[code]
    
    # For aux data codes
    if code == 0x3f:
        return 'aux_string'
    if code == 0x3e:
        return 'lev_change'
    if code == 0x39:
        return 'param'
    
    return f'unknown_{code}'


def read_dat_file(dat_path, fmt, n_samples, n_signals):
    """Read a .dat binary signal file."""
    fmt_map = {
        '212': (np.int16, 2, 1.5, 0.5),
        '16':  (np.int16, 2, 1.0, 0.0),
        '312': (np.int16, 2, 1.0, 0.0),
        '80':  (np.int8, 1, 1.0, 0.0),
        '24':  (np.uint8, 1, 1.0, 0.0),
    }
    
    if fmt in fmt_map:
        dtype, bytes_per_sample, gain_factor, offset = fmt_map[fmt]
    else:
        dtype = np.int16
        bytes_per_sample = 2
        gain_factor = 1.0
        offset = 0.0
    
    with open(dat_path, 'rb') as f:
        raw = f.read()
    
    if fmt == '212':
        # 212 format: 2 channels packed in 3 bytes
        # (2 bytes ch0, 1 byte ch1 low, ch1 high combined with ch0 low)
        n_samples_read = len(raw) // 3
        signals = np.zeros((n_samples_read, 2), dtype=np.float32)
        for i in range(n_samples_read):
            b0 = struct.unpack_from('<H', raw, i * 3)[0]
            b2 = raw[i * 3 + 2]
            ch0 = b0 >> 4 if b0 & 0x800 == 0 else (b0 >> 4) | 0xF000
            ch1 = b2 << 4 | (b0 & 0xF)
            if ch1 & 0x800:
                ch1 |= 0xF000
            signals[i, 0] = ch0
            signals[i, 1] = ch1
        return signals[:n_samples]
    else:
        # Generic interleaved read
        n_read = len(raw) // (bytes_per_sample * n_signals)
        raw_arr = np.frombuffer(raw, dtype=dtype)
        signals = raw_arr.reshape(n_read, n_signals).astype(np.float32)
        return signals[:n_samples] if n_samples else signals


def parse_incart_atr(atr_path):
    """
    Parse INCART annotation file.
    INCART uses similar format to MIT-BIH.
    """
    return parse_atr_annotation(atr_path, n_samples=None)


def parse_incart_patient_info(hea_comments):
    """Extract patient ID from INCART header comments."""
    for comment in hea_comments:
        if '# patient' in comment.lower():
            # "# patient 1" -> patient_id
            parts = comment.split()
            for i, p in enumerate(parts):
                if p.lower() == 'patient' and i + 1 < len(parts):
                    return parts[i + 1]
    return None


def get_mitbih_subject_mapping(hea_dir):
    """
    MIT-BIH: Map subjects to records.
    Records 201 and 202 belong to the same subject.
    Standard MIT-BIH subject mapping based on known record assignments.
    """
    # Known MIT-BIH subject mapping (from the database documentation)
    # Records with same 100-digit prefix or known to be same patient
    subject_map = {}
    records = sorted(glob.glob(os.path.join(hea_dir, '*.hea')))
    
    for hea in records:
        basename = os.path.basename(hea).replace('.hea', '')
        header = parse_hea_header(hea)
        record_num = int(basename) if basename.isdigit() else basename
        
        # MIT-BIH standard mapping:
        # Records 101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 223, 230, 231, 232, 233, 234
        # Records 100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234
        
        # Most records are unique patients. Exceptions:
        # 201 and 202 are the same patient
        # Some records share patients in the MIT-BIH database
        
        # Simple approach: use record number as subject for most,
        # but group 201 and 202 together
        subject_id = str(record_num)
        
        # Known same-patient records
        if record_num in (201, 202):
            subject_id = 'S201_202'
        elif record_num in (203, 215):
            subject_id = 'S203_215'  # Sometimes reported as same patient
        elif record_num in (231, 232):
            subject_id = 'S231_232'
        elif record_num in (233, 234):
            subject_id = 'S233_234'
        
        if subject_id not in subject_map:
            subject_map[subject_id] = []
        subject_map[subject_id].append(basename)
    
    return subject_map


def get_incart_patient_mapping(hea_dir):
    """
    INCART: Map patients to records based on WFDB header info.
    Records I01-I75, patient info in comments.
    """
    records = sorted(glob.glob(os.path.join(hea_dir, '*.hea')))
    patient_map = defaultdict(list)
    
    for hea in records:
        basename = os.path.basename(hea).replace('.hea', '')
        header = parse_hea_header(hea)
        
        # Try to extract patient ID from comments
        patient_id = parse_incart_patient_info(header['comments'])
        
        if patient_id is None:
            # Try to infer from record name - INCART uses I01-I75
            # but multiple records may belong to same patient
            # The comment should have "# patient N"
            # Fallback: use record name as patient
            patient_id = basename
        
        patient_map[patient_id].append(basename)
    
    return dict(patient_map)


# ============================================================
# AAMI Beat Annotation Mapping
# ============================================================
# AAMI EC57 standard mapping
AAMI_MAPPING = {
    # Normal beats (N)
    'N': 'N',  # Normal beat
    'L': 'N',  # Left bundle branch block
    'R': 'N',  # Right bundle branch block
    'e': 'N',  # Atrial escape beat
    'j': 'N',  # Nodal (junctional) escape beat
    
    # Supraventricular ectopic beats (S)
    'a': 'S',  # Atrial premature beat
    'A': 'S',  # Aberrated atrial premature beat
    'J': 'S',  # Nodal (junctional) premature beat
    'S': 'S',  # Supraventricular premature beat
    
    # Ventricular ectopic beats (V)
    'V': 'V',  # Premature ventricular contraction
    'E': 'V',  # Ventricular escape beat
    
    # Fusion beats (F)
    'F': 'F',  # Fusion of ventricular and normal beat
    
    # Unknown / Paced (Q)
    '/': 'Q',  # Paced beat
    'f': 'Q',  # Fusion of paced and normal beat
    'Q': 'Q',  # Unclassifiable beat
    
    # Ignored (non-beat annotations)
    '~': None,  # Signal quality change
    'x': None,  # Non-conducted P-wave
    '(': None,  # Waveform onset
    ')': None,  # Waveform end
    'p': None,  # P-wave peak
    't': None,  # T-wave peak
    'u': None,  # U-wave peak
    '`': None,  # MSYS peak
    "'": None,  # MSYS peak
    '^': None,  # Non-conducted pacemaker spike
    ',': None,  # Rhythm change
    'w': None,  # Wandering baseline
    'i': None,  # Electrode disconnect
    's': None,  # ST change
    'T': None,  # T-wave change
    '*': None,  # Heart rate change
    'D': None,  # Diastolic timing
    '"': None,  # Systolic timing
    '=': None,  # Measurement annotation
    '+': None,  # Measurement
    'P': 'Q',   # Pacer spike -> Q (unknown/paced)
}


def inspect_mitbih(hea_dir):
    """Inspect MIT-BIH Arrhythmia Database."""
    print("=" * 70)
    print("MIT-BIH ARRHYTHMIA DATABASE INSPECTION")
    print("=" * 70)
    
    records = sorted(glob.glob(os.path.join(hea_dir, '*.hea')))
    print(f"\nNumber of header files: {len(records)}")
    
    all_annotations = Counter()
    all_symbols = set()
    record_info = {}
    sampling_rates = set()
    
    for hea in records:
        basename = os.path.basename(hea).replace('.hea', '')
        header = parse_hea_header(hea)
        sampling_rates.add(header['fs'])
        
        # Read signal file to get actual length
        dat_path = hea.replace('.hea', '.dat')
        atr_path = hea.replace('.hea', '.atr')
        
        sig_names = [s['name'] for s in header['signals']]
        
        # Read annotations
        ann_count = Counter()
        symbols = set()
        if os.path.exists(atr_path):
            try:
                annotations = parse_atr_annotation(atr_path, header['n_samples'])
                for sample, sym in annotations:
                    if sym not in ('aux_string', 'lev_change', 'param'):
                        ann_count[sym] += 1
                        symbols.add(sym)
                all_symbols.update(symbols)
            except Exception as e:
                print(f"  Warning: Could not parse {atr_path}: {e}")
        
        record_info[basename] = {
            'fs': header['fs'],
            'n_signals': header['n_signals'],
            'signals': sig_names,
            'n_samples': header['n_samples'],
            'annotations': dict(ann_count),
        }
        all_annotations.update(ann_count)
        
        print(f"  {basename}: fs={header['fs']}Hz, {header['n_signals']} channels "
              f"[{', '.join(sig_names)}], {header['n_samples']} samples, "
              f"{sum(ann_count.values())} annotations")
    
    print(f"\nSampling rates: {sampling_rates}")
    print(f"\nAll annotation symbols: {sorted(all_symbols)}")
    print(f"\nAnnotation counts (raw):")
    for sym in sorted(all_annotations.keys()):
        mapped = AAMI_MAPPING.get(sym, '???')
        print(f"  '{sym}' -> '{mapped}': {all_annotations[sym]} beats")
    
    # Map to AAMI classes
    aami_counts = Counter()
    unmapped = Counter()
    for sym, count in all_annotations.items():
        mapped = AAMI_MAPPING.get(sym)
        if mapped is not None:
            aami_counts[mapped] += count
        else:
            unmapped[sym] += count
    
    print(f"\nAAMI class distribution:")
    for cls in ['N', 'S', 'V', 'F', 'Q']:
        print(f"  {cls}: {aami_counts.get(cls, 0)} beats")
    if unmapped:
        print(f"  Ignored: {sum(unmapped.values())} beats")
        for sym, count in unmapped.items():
            print(f"    '{sym}': {count}")
    
    # Subject mapping
    subject_map = get_mitbih_subject_mapping(hea_dir)
    print(f"\nSubject mapping ({len(subject_map)} subjects):")
    for subj, recs in sorted(subject_map.items()):
        print(f"  {subj}: {recs}")
    
    return record_info, all_annotations, subject_map


def inspect_incart(hea_dir):
    """Inspect St. Petersburg INCART Arrhythmia Database."""
    print("\n" + "=" * 70)
    print("ST. PETERSBURG INCART ARRHYTHMIA DATABASE INSPECTION")
    print("=" * 70)
    
    records = sorted(glob.glob(os.path.join(hea_dir, '*.hea')))
    print(f"\nNumber of header files: {len(records)}")
    
    all_annotations = Counter()
    all_symbols = set()
    record_info = {}
    sampling_rates = set()
    patient_ids = set()
    
    for hea in records:
        basename = os.path.basename(hea).replace('.hea', '')
        header = parse_hea_header(hea)
        sampling_rates.add(header['fs'])
        
        sig_names = [s['name'] for s in header['signals']]
        
        # Extract patient ID
        patient_id = parse_incart_patient_info(header['comments'])
        if patient_id:
            patient_ids.add(patient_id)
        
        atr_path = hea.replace('.hea', '.atr')
        
        ann_count = Counter()
        symbols = set()
        if os.path.exists(atr_path):
            try:
                annotations = parse_atr_annotation(atr_path, header['n_samples'])
                for sample, sym in annotations:
                    if sym not in ('aux_string', 'lev_change', 'param'):
                        ann_count[sym] += 1
                        symbols.add(sym)
                all_symbols.update(symbols)
            except Exception as e:
                print(f"  Warning: Could not parse {atr_path}: {e}")
        
        record_info[basename] = {
            'fs': header['fs'],
            'n_signals': header['n_signals'],
            'signals': sig_names,
            'n_samples': header['n_samples'],
            'annotations': dict(ann_count),
            'patient_id': patient_id,
        }
        all_annotations.update(ann_count)
        
        print(f"  {basename}: fs={header['fs']}Hz, {header['n_signals']} channels "
              f"[{', '.join(sig_names[:4])}{'...' if len(sig_names) > 4 else ''}], "
              f"patient={patient_id}, {sum(ann_count.values())} annotations")
    
    print(f"\nSampling rates: {sampling_rates}")
    print(f"Unique patients: {len(patient_ids)} - {sorted(patient_ids)}")
    print(f"\nAll annotation symbols: {sorted(all_symbols)}")
    print(f"\nAnnotation counts (raw):")
    for sym in sorted(all_annotations.keys()):
        mapped = AAMI_MAPPING.get(sym, '???')
        print(f"  '{sym}' -> '{mapped}': {all_annotations[sym]} beats")
    
    # Map to AAMI classes
    aami_counts = Counter()
    unmapped = Counter()
    for sym, count in all_annotations.items():
        mapped = AAMI_MAPPING.get(sym)
        if mapped is not None:
            aami_counts[mapped] += count
        else:
            unmapped[sym] += count
    
    print(f"\nAAMI class distribution:")
    for cls in ['N', 'S', 'V', 'F', 'Q']:
        print(f"  {cls}: {aami_counts.get(cls, 0)} beats")
    if unmapped:
        print(f"  Ignored: {sum(unmapped.values())} beats")
        for sym, count in unmapped.items():
            print(f"    '{sym}': {count}")
    
    # Patient mapping
    patient_map = get_incart_patient_mapping(hea_dir)
    print(f"\nPatient mapping ({len(patient_map)} patients):")
    for pat, recs in sorted(patient_map.items()):
        print(f"  {pat}: {recs}")
    
    return record_info, all_annotations, patient_map


if __name__ == '__main__':
    mit_dir = 'd:/git/mamintoosi-cs/holter-ecg-analysis/data/mit-bih'
    incart_dir = 'd:/git/mamintoosi-cs/holter-ecg-analysis/data/incartdb'
    
    print("Inspecting MIT-BIH...")
    mit_info, mit_ann, mit_subjects = inspect_mitbih(mit_dir)
    
    print("\n\nInspecting INCART...")
    incart_info, incart_ann, incart_patients = inspect_incart(incart_dir)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"MIT-BIH: {len(mit_info)} records, {sum(mit_ann.values())} total annotations")
    print(f"INCART: {len(incart_info)} records, {sum(incart_ann.values())} total annotations")
