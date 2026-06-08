from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import torch

from .config import NRPhyConfig
from .dsp import delay_domain_denoise_csi
from .models.comm_foundation_model import (
    CommFoundationChannelEstimator,
    CommFoundationConfig,
    channels_to_complex_np,
    complex_np_to_channels,
)
from .utils import ComplexArray


class LearnedChannelEstimator:
    """Inference wrapper for the complex communication foundation estimator."""

    def __init__(
        self,
        checkpoint_path: str,
        phy_cfg: NRPhyConfig,
        device: str | None = None,
        use_delay_denoise: bool = True,
        delay_denoise_taps: int | None = None,
        time_average: bool = False,
    ) -> None:
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(checkpoint_path)
        self.phy_cfg = phy_cfg
        self.use_delay_denoise = use_delay_denoise
        self.delay_denoise_taps = delay_denoise_taps
        self.time_average = time_average
        self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        payload = torch.load(checkpoint_path, map_location=self.device)
        cfg = CommFoundationConfig(**payload["model_config"])
        self.model = CommFoundationChannelEstimator(cfg).to(self.device)
        self.model.load_state_dict(payload["state_dict"], strict=False)
        self.model.eval()
        self.last_inference_time_ms = 0.0

    @torch.no_grad()
    def predict(self, h_ls_grid: ComplexArray) -> ComplexArray:
        h_input = np.asarray(h_ls_grid, dtype=np.complex128)
        if self.use_delay_denoise:
            h_input = delay_domain_denoise_csi(
                h_input,
                self.phy_cfg,
                n_taps=self.delay_denoise_taps,
                time_average=self.time_average,
            )
        x = complex_np_to_channels(np.asarray(h_input, dtype=np.complex64)[None]).to(self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        pred = self.model(x)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.last_inference_time_ms = (time.perf_counter() - start) * 1000.0
        return channels_to_complex_np(pred)[0].astype(np.complex128)
