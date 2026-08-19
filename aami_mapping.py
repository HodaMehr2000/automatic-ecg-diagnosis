"""
aami_mapping.py
AAMI EC57-style beat annotation mapping for MIT-BIH and INCART databases.

Classes:
  N = Normal and bundle-branch-related beats
  S = Supraventricular ectopic beats
  V = Ventricular ectopic beats
  F = Fusion beats
  Q = Unknown/paced/unclassifiable beats
"""
from collections import Counter

# ============================================================
# AAMI Beat Classification Mapping (AAMI EC57 standard)
# ============================================================
# Maps original WFDB annotation symbols to AAMI classes.
# None means the annotation is NOT a beat (e.g., rhythm change,
# T-wave marker, signal quality) and should be SKIPPED.

AAMI_MAP = {
    # ---- N class: Normal / bundle-branch ----
    'N': 'N',   # Normal beat
    'L': 'N',   # Left bundle branch block beat
    'R': 'N',   # Right bundle branch block beat
    'e': 'N',   # Atrial escape beat
    'j': 'N',   # Nodal (junctional) escape beat
    'n': 'N',   # Normal beat (anonymized, used in INCART)

    # ---- S class: Supraventricular ectopic ----
    'a': 'S',   # Atrial premature beat
    'A': 'S',   # Aberrated atrial premature beat
    'J': 'S',   # Nodal (junctional) premature beat
    'S': 'S',   # Supraventricular premature beat

    # ---- V class: Ventricular ectopic ----
    'V': 'V',   # Premature ventricular contraction
    'E': 'V',   # Ventricular escape beat

    # ---- F class: Fusion ----
    'F': 'F',   # Fusion of ventricular and normal beat

    # ---- Q class: Unknown / paced / unclassifiable ----
    '/': 'Q',   # Paced beat
    'f': 'Q',   # Fusion of paced and normal beat
    'Q': 'Q',   # Unclassifiable beat
    'B': 'Q',   # Bursted/paced beat (INCART-specific?)

    # ---- Ignored (non-beat annotations) ----
    '+': None,   # Measurement/rhythm change annotation
    '~': None,   # Signal quality change
    '|': None,   # Segment separator
    '=': None,   # Measurement annotation
    'x': None,   # Non-conducted P-wave (not a beat)
    '(': None,   # Waveform onset
    ')': None,   # Waveform end
    'p': None,   # P-wave peak
    't': None,   # T-wave peak
    'u': None,   # U-wave peak
    '`': None,   # Peak of N complex
    "'": None,   # Peak of N complex
    '^': None,   # Non-conducted pacemaker spike
    ',': None,   # Rhythm change
    'w': None,   # Aberrated/anterior misaligned QRS
    'i': None,   # Electrode fall-off
    's': None,   # ST change
    'T': None,   # T-wave change
    '*': None,   # Heart rate change
    'D': None,   # Diastolic timing
    '"': None,   # Systolic timing
    '[': None,   # Start of ventricular flutter/fibrillation
    ']': None,   # End of ventricular flutter/fibrillation
    '!': None,   # Ventricular flutter wave
}

# AAMI class names in order
AAMI_CLASSES = ['N', 'S', 'V', 'F', 'Q']


def map_symbol(symbol):
    """Map a WFDB annotation symbol to an AAMI class.
    Returns None if the symbol is not a beat annotation."""
    return AAMI_MAP.get(symbol)


def map_symbols(symbols):
    """Map a list of WFDB symbols to AAMI classes, filtering out non-beats.
    Returns (mapped_labels, original_indices)."""
    labels = []
    indices = []
    for i, sym in enumerate(symbols):
        aami = map_symbol(sym)
        if aami is not None:
            labels.append(aami)
            indices.append(i)
    return labels, indices


def print_mapping_report():
    """Print the full mapping table and verify completeness."""
    print("=" * 60)
    print("AAMI EC57 BEAT ANNOTATION MAPPING")
    print("=" * 60)
    print(f"\n{'Original':>12} -> {'AAMI Class':>12}  Description")
    print("-" * 60)
    
    descriptions = {
        'N': 'Normal beat',
        'L': 'Left bundle branch block',
        'R': 'Right bundle branch block',
        'e': 'Atrial escape beat',
        'j': 'Nodal escape beat',
        'n': 'Normal beat (anonymized)',
        'a': 'Atrial premature beat',
        'A': 'Aberrated atrial premature',
        'J': 'Nodal premature beat',
        'S': 'Supraventricular premature',
        'V': 'Premature ventricular contraction',
        'E': 'Ventricular escape beat',
        'F': 'Fusion (V + normal)',
        '/': 'Paced beat',
        'f': 'Fusion (paced + normal)',
        'Q': 'Unclassifiable beat',
        'B': 'Burst/paced beat',
    }
    
    for sym in sorted(AAMI_MAP.keys()):
        aami = AAMI_MAP[sym]
        desc = descriptions.get(sym, '(non-beat, ignored)')
        print(f"  {sym:>10} -> {str(aami):>12}  {desc}")
    
    print(f"\nAAMI classes: {AAMI_CLASSES}")
    print(f"  N = Normal (bundle-branch included)")
    print(f"  S = Supraventricular ectopic")
    print(f"  V = Ventricular ectopic")
    print(f"  F = Fusion")
    print(f"  Q = Unknown/Paced/Unclassifiable")


if __name__ == '__main__':
    print_mapping_report()
