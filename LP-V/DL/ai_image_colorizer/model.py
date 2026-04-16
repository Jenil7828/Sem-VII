from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Two stacked conv blocks used throughout U-Net."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = nn.functional.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNetColorizer(nn.Module):
    """U-Net for colorization: L channel -> (a, b) channels."""

    def __init__(self, in_channels: int = 1, out_channels: int = 2, base_channels: int = 32) -> None:
        super().__init__()
        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)

        self.up1 = Up(base_channels * 8 + base_channels * 4, base_channels * 4)
        self.up2 = Up(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.up3 = Up(base_channels * 2 + base_channels, base_channels)

        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

        # Default warm-tone bias so non-pretrained inference remains visually meaningful.
        nn.init.zeros_(self.out_conv.weight)
        nn.init.constant_(self.out_conv.bias[0:1], 0.03)
        nn.init.constant_(self.out_conv.bias[1:2], 0.08)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        out = torch.tanh(self.out_conv(x))

        if not return_features:
            return out

        features = {
            "enc1": x1,
            "enc2": x2,
            "enc3": x3,
            "bottleneck": x4,
        }
        return out, features


def load_unet_model(weights_path: Path, device: torch.device) -> Tuple[UNetColorizer, bool]:
    """Load U-Net model and optional trained checkpoint."""
    model = UNetColorizer().to(device)
    loaded_pretrained = False

    if weights_path.exists():
        state = torch.load(weights_path, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=True)
        loaded_pretrained = True

    model.eval()
    return model, loaded_pretrained


def summarize_model(model: nn.Module) -> str:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return (
        f"Model: UNetColorizer\\n"
        f"Total parameters: {total_params:,}\\n"
        f"Trainable parameters: {trainable_params:,}\\n"
        f"Input: (B, 1, H, W) grayscale L channel\\n"
        f"Output: (B, 2, H, W) predicted a,b channels"
    )
