from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from .complex_layers import ComplexConv2d, ComplexLayerNorm, ComplexReLU


@dataclass
class CommFoundationConfig:
    in_complex_channels: int = 1
    hidden_complex_channels: int = 16
    depth: int = 4
    residual_scale: float = 1.0


class ComplexCommunicationBackbone(nn.Module):
    """Lightweight complex CNN backbone that outputs Z_comm."""

    def __init__(self, cfg: CommFoundationConfig):
        super().__init__()
        layers = []
        in_ch = cfg.in_complex_channels
        for _ in range(cfg.depth):
            layers.extend(
                [
                    ComplexConv2d(in_ch, cfg.hidden_complex_channels, kernel_size=3, padding=1),
                    ComplexLayerNorm(cfg.hidden_complex_channels),
                    ComplexReLU(),
                ]
            )
            in_ch = cfg.hidden_complex_channels
        self.net = nn.Sequential(*layers)
        self.out_complex_channels = cfg.hidden_complex_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ChannelEstimationHead(nn.Module):
    """Maps Z_comm to a residual channel correction with shape [B, 2, F, T]."""

    def __init__(self, in_complex_channels: int, zero_init: bool = True):
        super().__init__()
        self.proj = ComplexConv2d(in_complex_channels, 1, kernel_size=1, padding=0)
        if zero_init:
            nn.init.zeros_(self.proj.real.weight)
            nn.init.zeros_(self.proj.imag.weight)
            if self.proj.real.bias is not None:
                nn.init.zeros_(self.proj.real.bias)
            if self.proj.imag.bias is not None:
                nn.init.zeros_(self.proj.imag.bias)

    def forward(self, z_comm: torch.Tensor) -> torch.Tensor:
        return self.proj(z_comm)


class CSIFeedbackHead(nn.Module):
    """Placeholder for future learned CSI feedback compression/reconstruction."""

    def __init__(self, in_complex_channels: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, z_comm: torch.Tensor) -> torch.Tensor:
        return self.pool(z_comm).flatten(1)


class ReliabilityHead(nn.Module):
    """Predict a [0, 1] time-frequency reliability map aligned with the grid."""

    def __init__(self, in_complex_channels: int):
        super().__init__()
        self.proj = nn.Conv2d(2 * in_complex_channels, 1, kernel_size=1)

    def forward(self, z_comm: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.proj(z_comm))


class SemanticAssistHead(nn.Module):
    """Placeholder head for future SwinJSCC decoder assistance features."""

    def __init__(self, in_complex_channels: int, assist_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(2 * in_complex_channels, assist_dim))

    def forward(self, z_comm: torch.Tensor) -> torch.Tensor:
        return self.net(z_comm)


class CommFoundationChannelEstimator(nn.Module):
    def __init__(self, cfg: CommFoundationConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = ComplexCommunicationBackbone(cfg)
        self.channel_head = ChannelEstimationHead(self.backbone.out_complex_channels)
        self.residual_scale_param = nn.Parameter(torch.tensor(float(cfg.residual_scale)))
        self.csi_feedback_head = CSIFeedbackHead(self.backbone.out_complex_channels)
        self.reliability_head = ReliabilityHead(self.backbone.out_complex_channels)
        self.semantic_assist_head = SemanticAssistHead(self.backbone.out_complex_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_comm = self.backbone(x)
        residual = self.channel_head(z_comm)
        return x[:, :2] + self.residual_scale_param * residual

    def residual(self, x: torch.Tensor) -> torch.Tensor:
        z_comm = self.backbone(x)
        return self.channel_head(z_comm)

    def z_comm(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def checkpoint_payload(self) -> dict:
        return {
            "model_config": asdict(self.cfg),
            "state_dict": self.state_dict(),
        }


def complex_np_to_channels(x) -> torch.Tensor:
    real = torch.from_numpy(x.real).float()
    imag = torch.from_numpy(x.imag).float()
    if real.ndim == 3:
        real = real[:, None]
        imag = imag[:, None]
    return torch.cat([real, imag], dim=1)


def channels_to_complex_np(x: torch.Tensor):
    x = x.detach().cpu()
    half = x.size(1) // 2
    arr = x[:, :half].numpy() + 1j * x[:, half:].numpy()
    return arr.squeeze(1)


def nmse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    err = torch.mean((pred - target) ** 2, dim=(1, 2, 3))
    power = torch.mean(target**2, dim=(1, 2, 3)).clamp_min(1e-12)
    return torch.mean(err / power)
