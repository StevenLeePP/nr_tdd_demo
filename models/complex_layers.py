from __future__ import annotations

import torch
from torch import nn


def split_complex(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if x.size(1) % 2 != 0:
        raise ValueError("Complex tensors use channel layout [real..., imag...].")
    half = x.size(1) // 2
    return x[:, :half], x[:, half:]


def join_complex(real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
    return torch.cat([real, imag], dim=1)


class ComplexConv2d(nn.Module):
    """Complex 2D convolution for [B, 2*C, H, W] tensors."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size=3, padding=1, bias=True):
        super().__init__()
        self.real = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)
        self.imag = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xr, xi = split_complex(x)
        yr = self.real(xr) - self.imag(xi)
        yi = self.real(xi) + self.imag(xr)
        return join_complex(yr, yi)


class ComplexLinear(nn.Module):
    """Complex linear layer for [B, 2*C] tensors."""

    def __init__(self, in_features: int, out_features: int, bias=True):
        super().__init__()
        self.real = nn.Linear(in_features, out_features, bias=bias)
        self.imag = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xr, xi = split_complex(x.unsqueeze(-1).unsqueeze(-1))
        xr = xr.flatten(1)
        xi = xi.flatten(1)
        yr = self.real(xr) - self.imag(xi)
        yi = self.real(xi) + self.imag(xr)
        return torch.cat([yr, yi], dim=1)


class ComplexReLU(nn.Module):
    """Applies ReLU separately to real and imaginary components."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xr, xi = split_complex(x)
        return join_complex(torch.relu(xr), torch.relu(xi))


class ComplexLayerNorm(nn.Module):
    """Simple complex layer norm over real and imaginary channels."""

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.norm = nn.GroupNorm(1, 2 * channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)

