"""
Modernized PyTorch ResNet1D for ECG classification.

Architecture from:
    https://github.com/antonior92/ecg-age-prediction
    which implements the model described in:
    Ribeiro et al., "Automatic diagnosis of the 12-lead ECG using a deep neural network",
    Nature Communications 11, 1760 (2020).

Updated for PyTorch >= 2.0 with modern APIs.
"""

import torch
import torch.nn as nn


def _padding(downsample, kernel_size):
    """Compute required padding."""
    padding = max(0, int((kernel_size - downsample + 1) // 2))
    return padding


def _downsample(n_samples_in, n_samples_out):
    """Compute downsample rate."""
    downsample = int(n_samples_in // n_samples_out)
    if downsample < 1:
        raise ValueError("Number of samples should always decrease")
    if n_samples_in % n_samples_out != 0:
        raise ValueError(
            "Number of samples for two consecutive blocks "
            "should always decrease by an integer factor."
        )
    return downsample


class ResBlock1d(nn.Module):
    """Residual network unit for unidimensional signals."""

    def __init__(self, n_filters_in, n_filters_out, downsample, kernel_size, dropout_rate):
        if kernel_size % 2 == 0:
            raise ValueError(
                "The current implementation only supports odd values for `kernel_size`."
            )
        super().__init__()

        # Forward path
        padding = _padding(1, kernel_size)
        self.conv1 = nn.Conv1d(
            n_filters_in, n_filters_out, kernel_size, padding=padding, bias=False
        )
        self.bn1 = nn.BatchNorm1d(n_filters_out)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)

        padding = _padding(downsample, kernel_size)
        self.conv2 = nn.Conv1d(
            n_filters_out,
            n_filters_out,
            kernel_size,
            stride=downsample,
            padding=padding,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(n_filters_out)
        self.dropout2 = nn.Dropout(dropout_rate)

        # Skip connection
        skip_connection_layers = []
        if downsample > 1:
            maxpool = nn.MaxPool1d(downsample, stride=downsample)
            skip_connection_layers.append(maxpool)
        if n_filters_in != n_filters_out:
            conv1x1 = nn.Conv1d(n_filters_in, n_filters_out, 1, bias=False)
            skip_connection_layers.append(conv1x1)

        if skip_connection_layers:
            self.skip_connection = nn.Sequential(*skip_connection_layers)
        else:
            self.skip_connection = None

    def forward(self, x, y):
        """Residual unit."""
        if self.skip_connection is not None:
            y = self.skip_connection(y)

        # 1st layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        # 2nd layer
        x = self.conv2(x)
        x += y  # Sum skip connection and main connection
        y = x
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)

        return x, y


class ResNet1d(nn.Module):
    """
    Residual network for unidimensional signals.

    Parameters
    ----------
    input_dim : tuple
        Input dimensions: ``(n_channels, n_samples)``.
        For ECG: ``(12, 4096)``.
    blocks_dim : list of tuples
        Dimensions of residual blocks. Each tuple: ``(n_filters, n_samples)``.
    n_classes : int
        Number of output classes.
    kernel_size : int, optional
        Kernel size for convolutional layers (must be odd). Default is 17.
    dropout_rate : float, optional
        Dropout rate. Default is 0.8.
    """

    def __init__(self, input_dim, blocks_dim, n_classes, kernel_size=17, dropout_rate=0.8):
        super().__init__()

        # First layers
        n_filters_in, n_filters_out = input_dim[0], blocks_dim[0][0]
        n_samples_in, n_samples_out = input_dim[1], blocks_dim[0][1]
        downsample = _downsample(n_samples_in, n_samples_out)
        padding = _padding(downsample, kernel_size)

        self.conv1 = nn.Conv1d(
            n_filters_in, n_filters_out, kernel_size, bias=False,
            stride=downsample, padding=padding
        )
        self.bn1 = nn.BatchNorm1d(n_filters_out)

        # Residual block layers
        self.res_blocks = nn.ModuleList()
        for i, (n_filters, n_samples) in enumerate(blocks_dim):
            n_filters_in, n_filters_out = n_filters_out, n_filters
            n_samples_in, n_samples_out = n_samples_out, n_samples
            downsample = _downsample(n_samples_in, n_samples_out)
            resblk = ResBlock1d(
                n_filters_in, n_filters_out, downsample, kernel_size, dropout_rate
            )
            self.res_blocks.append(resblk)

        # Linear layer
        n_filters_last, n_samples_last = blocks_dim[-1]
        last_layer_dim = n_filters_last * n_samples_last
        self.lin = nn.Linear(last_layer_dim, n_classes)
        self.n_blk = len(blocks_dim)

    def forward(self, x):
        """Forward pass.

        Args:
            x: Tensor of shape (batch, n_channels, n_samples).

        Returns:
            Tensor of shape (batch, n_classes) with raw logits.
        """
        # First layers
        x = self.conv1(x)
        x = self.bn1(x)

        # Residual blocks
        y = x
        for blk in self.res_blocks:
            x, y = blk(x, y)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layer
        x = self.lin(x)

        return x


def build_ecg_resnet(n_classes=6, kernel_size=17, dropout_rate=0.8, device=None):
    """
    Build the standard ECG ResNet1D model.

    Default architecture matches the original paper:
        input: (batch, 12, 4096)
        blocks: [(64, 4096), (128, 1024), (196, 256), (256, 64), (320, 16)]
        output: (batch, n_classes) logits

    Args:
        n_classes: Number of output classes.
        kernel_size: Convolution kernel size.
        dropout_rate: Dropout rate.
        device: Device to move model to (optional).

    Returns:
        ResNet1d model instance.
    """
    N_LEADS = 12
    SEQ_LENGTH = 4096
    NET_FILTER_SIZE = [64, 128, 196, 256, 320]
    NET_SEQ_LENGTH = [4096, 1024, 256, 64, 16]

    model = ResNet1d(
        input_dim=(N_LEADS, SEQ_LENGTH),
        blocks_dim=list(zip(NET_FILTER_SIZE, NET_SEQ_LENGTH)),
        n_classes=n_classes,
        kernel_size=kernel_size,
        dropout_rate=dropout_rate,
    )

    if device is not None:
        model = model.to(device)

    return model


if __name__ == "__main__":
    # Quick sanity check
    model = build_ecg_resnet(n_classes=6)
    x = torch.randn(2, 12, 4096)
    out = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Output (logits):\n{out}")
    print(f"\nModel parameter count: {sum(p.numel() for p in model.parameters()):,}")
