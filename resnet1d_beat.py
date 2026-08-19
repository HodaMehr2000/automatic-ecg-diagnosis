"""
resnet1d_beat.py
Lightweight 1D ResNet for single-lead beat classification.
Designed to be CPU-friendly with small filter counts.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock1D(nn.Module):
    """Basic residual block for 1D signals."""
    
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1, dropout=0.2):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet1D_Beat(nn.Module):
    """
    Lightweight ResNet1D for single-lead ECG beat classification.
    
    Architecture:
    - Initial conv: 1 -> 16 channels
    - 3 residual blocks: 16 -> 32 -> 64 -> 128
    - Global average pooling
    - FC: 128 -> 5 classes
    
    For 250 Hz, 400-sample input (1.6s beat window):
    - After conv stride 2: 200 samples
    - After block 1 stride 2: 100 samples
    - After block 2 stride 2: 50 samples
    - After block 3 stride 2: 25 samples
    - Global avg pool: 128-dim vector
    - FC -> 5 classes
    
    Total params: ~250K (very lightweight)
    """
    
    def __init__(self, n_classes=5, in_channels=1, dropout=0.3):
        super().__init__()
        
        # Initial convolution
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
        )
        
        # Residual blocks
        self.block1 = ResBlock1D(16, 32, kernel_size=7, stride=2, dropout=dropout)
        self.block2 = ResBlock1D(32, 64, kernel_size=7, stride=2, dropout=dropout)
        self.block3 = ResBlock1D(64, 128, kernel_size=7, stride=2, dropout=dropout)
        
        # Global average pooling + classifier
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, n_classes),
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, 1, beat_len) single-lead ECG beat
        Returns:
            logits: (batch, n_classes)
        """
        x = self.stem(x)     # (B, 16, beat_len/2)
        x = self.block1(x)   # (B, 32, beat_len/4)
        x = self.block2(x)   # (B, 64, beat_len/8)
        x = self.block3(x)   # (B, 128, beat_len/16)
        x = self.gap(x)      # (B, 128, 1)
        x = x.squeeze(-1)    # (B, 128)
        logits = self.classifier(x)  # (B, n_classes)
        return logits


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Quick test
    model = ResNet1D_Beat(n_classes=5)
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Test forward pass with expected input shape
    x = torch.randn(4, 1, 400)  # batch=4, 1 channel, 400 samples (250Hz * 1.6s)
    logits = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Output sample: {logits[0].tolist()}")
