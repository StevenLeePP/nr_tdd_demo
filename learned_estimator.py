from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import torch

from .models.comm_foundation_model import (
    CommFoundationChannelEstimator,
    CommFoundationConfig,
    channels_to_complex_np,
    complex_np_to_channels,
)
from .utils import ComplexArray


class LearnedChannelEstimator:
    """Inference wrapper for the complex communication foundation estimator."""

    def __init__(self, checkpoint_path: str, device: str | None = None) -> None:
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(checkpoint_path)
        self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        payload = torch.load(checkpoint_path, map_location=self.device)
        cfg = CommFoundationConfig(**payload["model_config"])
        self.model = CommFoundationChannelEstimator(cfg).to(self.device)
        self.model.load_state_dict(payload["state_dict"], strict=True)
        self.model.eval()
        self.last_inference_time_ms = 0.0

    @torch.no_grad()
    def predict(self, h_ls_grid: ComplexArray) -> ComplexArray:
        x = complex_np_to_channels(np.asarray(h_ls_grid, dtype=np.complex64)[None]).to(self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        pred = self.model(x)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.last_inference_time_ms = (time.perf_counter() - start) * 1000.0
        return channels_to_complex_np(pred)[0].astype(np.complex128)
