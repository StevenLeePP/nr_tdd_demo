from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .config import NRPhyConfig
from .utils import ComplexArray, db_to_linear


@dataclass
class ChannelOutput:
    waveform: ComplexArray
    impulse_response: ComplexArray
    noise_variance: float
    measured_snr_db: float


class MultipathChannel:
    """AWGN, Rayleigh TDL, and Rician TDL channel model."""

    def __init__(
        self,
        cfg: NRPhyConfig,
        channel_type: str = "rayleigh",
        delays: Sequence[int] = (0, 2, 5),
        powers_db: Sequence[float] = (0.0, -3.0, -8.0),
        rician_k_db: float = 6.0,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.cfg = cfg
        self.channel_type = channel_type.lower()
        self.delays = np.asarray(delays, dtype=int)
        self.powers_db = np.asarray(powers_db, dtype=float)
        self.powers_linear = db_to_linear(self.powers_db)
        self.powers_linear = self.powers_linear / np.sum(self.powers_linear)
        self.rician_k_linear = db_to_linear(rician_k_db)
        self.rng = rng if rng is not None else np.random.default_rng(cfg.rng_seed)

        if self.delays.size != self.powers_linear.size:
            raise ValueError("delays and powers_db must have the same length.")
        if np.any(self.delays < 0):
            raise ValueError("Path delays must be non-negative sample offsets.")
        if self.channel_type not in {"awgn", "rayleigh", "rician"}:
            raise ValueError("channel_type must be awgn, rayleigh, or rician.")

    def describe(self) -> str:
        return (
            f"{self.channel_type.upper()} channel, target SNR {self.cfg.snr_db:g} dB, "
            f"delays={self.delays.tolist()}, PDP(dB)={self.powers_db.tolist()}"
        )

    def sample_impulse_response(self) -> ComplexArray:
        if self.channel_type == "awgn":
            return np.array([1.0 + 0.0j], dtype=np.complex128)

        taps = (
            self.rng.normal(size=self.delays.size)
            + 1j * self.rng.normal(size=self.delays.size)
        ) / math.sqrt(2.0)
        taps *= np.sqrt(self.powers_linear)

        if self.channel_type == "rician":
            k = self.rician_k_linear
            los_power = self.powers_linear[0]
            diffuse = taps[0] * math.sqrt(1.0 / (k + 1.0))
            los = math.sqrt(k / (k + 1.0) * los_power)
            taps[0] = los + diffuse

        h = np.zeros(int(np.max(self.delays)) + 1, dtype=np.complex128)
        for delay, tap in zip(self.delays, taps):
            h[delay] += tap
        return h

    def transmit(self, waveform: ComplexArray, h: Optional[ComplexArray] = None) -> ChannelOutput:
        h = self.sample_impulse_response() if h is None else np.asarray(h, dtype=np.complex128)

        # Time-domain convolution makes the TDL channel frequency-selective
        # across OFDM subcarriers once the receiver FFTs each CP-protected symbol.
        faded = np.convolve(waveform, h, mode="full")[: waveform.size]
        signal_power = float(np.mean(np.abs(faded) ** 2))
        noise_variance = signal_power / db_to_linear(self.cfg.snr_db)
        noise = math.sqrt(noise_variance / 2.0) * (
            self.rng.normal(size=faded.size) + 1j * self.rng.normal(size=faded.size)
        )
        measured_snr_db = 10.0 * math.log10(signal_power / max(noise_variance, 1e-30))
        return ChannelOutput(faded + noise, h, noise_variance, measured_snr_db)
