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
    frequency_response_grid: ComplexArray | None = None


class MultipathChannel:
    """AWGN, Rayleigh TDL, and Rician TDL channel model."""

    def __init__(
        self,
        cfg: NRPhyConfig,
        channel_type: str = "rayleigh",
        delays: Sequence[int] = (0, 2, 5),
        powers_db: Sequence[float] = (0.0, -3.0, -8.0),
        rician_k_db: float = 6.0,
        doppler_hz: float = 0.0,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.cfg = cfg
        self.channel_type = channel_type.lower()
        self.delays = np.asarray(delays, dtype=int)
        self.powers_db = np.asarray(powers_db, dtype=float)
        self.powers_linear = db_to_linear(self.powers_db)
        self.powers_linear = self.powers_linear / np.sum(self.powers_linear)
        self.rician_k_linear = db_to_linear(rician_k_db)
        self.doppler_hz = float(doppler_hz)
        self.rng = rng if rng is not None else np.random.default_rng(cfg.rng_seed)

        if self.delays.size != self.powers_linear.size:
            raise ValueError("delays and powers_db must have the same length.")
        if np.any(self.delays < 0):
            raise ValueError("Path delays must be non-negative sample offsets.")
        if self.channel_type not in {"awgn", "rayleigh", "rician"}:
            raise ValueError("channel_type must be awgn, rayleigh, or rician.")
        if self.doppler_hz < 0:
            raise ValueError("doppler_hz must be non-negative.")

    def describe(self) -> str:
        return (
            f"{self.channel_type.upper()} channel, target SNR {self.cfg.snr_db:g} dB, "
            f"delays={self.delays.tolist()}, PDP(dB)={self.powers_db.tolist()}, "
            f"doppler={self.doppler_hz:g} Hz"
        )

    def sample_impulse_response(self) -> ComplexArray:
        if self.cfg.is_mimo:
            return self.sample_mimo_impulse_response()
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

    def _sample_taps(self) -> ComplexArray:
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

    def sample_mimo_impulse_response(self) -> ComplexArray:
        if self.channel_type == "awgn":
            h = np.zeros((self.cfg.num_rx_antennas, self.cfg.num_tx_antennas, 1), dtype=np.complex128)
            for ant_idx in range(min(self.cfg.num_rx_antennas, self.cfg.num_tx_antennas)):
                h[ant_idx, ant_idx, 0] = 1.0 + 0.0j
            return h
        tap_len = int(np.max(self.delays)) + 1
        h = np.zeros((self.cfg.num_rx_antennas, self.cfg.num_tx_antennas, tap_len), dtype=np.complex128)
        for rx_idx in range(self.cfg.num_rx_antennas):
            for tx_idx in range(self.cfg.num_tx_antennas):
                h[rx_idx, tx_idx] = self._sample_taps()
        return h

    def transmit(self, waveform: ComplexArray, h: Optional[ComplexArray] = None) -> ChannelOutput:
        h = self.sample_impulse_response() if h is None else np.asarray(h, dtype=np.complex128)
        waveform = np.asarray(waveform, dtype=np.complex128)
        if waveform.ndim == 2 or h.ndim == 3:
            return self._transmit_mimo(waveform, h)

        if self.doppler_hz > 0.0 and self.channel_type != "awgn" and h.size > 1:
            faded, h_grid = self._transmit_time_varying(waveform, h)
        else:
            # Time-domain convolution makes the TDL channel frequency-selective
            # across OFDM subcarriers once the receiver FFTs each CP-protected symbol.
            faded = np.convolve(waveform, h, mode="full")[: waveform.size]
            n_slots = max(1, int(round(waveform.size / self.cfg.slot_samples)))
            h_grid = impulse_response_to_grid(self.cfg, h, n_slots=n_slots)
        signal_power = float(np.mean(np.abs(faded) ** 2))
        noise_variance = signal_power / db_to_linear(self.cfg.snr_db)
        noise = math.sqrt(noise_variance / 2.0) * (
            self.rng.normal(size=faded.size) + 1j * self.rng.normal(size=faded.size)
        )
        measured_snr_db = 10.0 * math.log10(signal_power / max(noise_variance, 1e-30))
        return ChannelOutput(faded + noise, h, noise_variance, measured_snr_db, h_grid)

    def _transmit_mimo(self, waveform: ComplexArray, h: ComplexArray) -> ChannelOutput:
        if waveform.ndim != 2:
            waveform = np.tile(waveform[None, :], (self.cfg.num_tx_antennas, 1))
        if h.ndim != 3:
            h_mimo = np.zeros((self.cfg.num_rx_antennas, self.cfg.num_tx_antennas, h.size), dtype=np.complex128)
            for rx_idx in range(self.cfg.num_rx_antennas):
                for tx_idx in range(self.cfg.num_tx_antennas):
                    h_mimo[rx_idx, tx_idx] = h
            h = h_mimo
        faded = np.zeros((h.shape[0], waveform.shape[1]), dtype=np.complex128)
        for rx_idx in range(h.shape[0]):
            for tx_idx in range(h.shape[1]):
                faded[rx_idx] += np.convolve(waveform[tx_idx], h[rx_idx, tx_idx], mode="full")[: waveform.shape[1]]
        n_slots = max(1, int(round(waveform.shape[1] / self.cfg.slot_samples)))
        h_grid = mimo_impulse_response_to_grid(self.cfg, h, n_slots=n_slots)
        signal_power = float(np.mean(np.abs(faded) ** 2))
        noise_variance = signal_power / db_to_linear(self.cfg.snr_db)
        noise = math.sqrt(noise_variance / 2.0) * (
            self.rng.normal(size=faded.shape) + 1j * self.rng.normal(size=faded.shape)
        )
        measured_snr_db = 10.0 * math.log10(signal_power / max(noise_variance, 1e-30))
        return ChannelOutput(faded + noise, h, noise_variance, measured_snr_db, h_grid)

    def _transmit_time_varying(self, waveform: ComplexArray, h: ComplexArray) -> tuple[ComplexArray, ComplexArray]:
        active_delays = np.flatnonzero(np.abs(h) > 0)
        taps = h[active_delays]
        path_dopplers = self.rng.uniform(-self.doppler_hz, self.doppler_hz, size=active_delays.size)
        n = np.arange(waveform.size, dtype=float)
        faded = np.zeros_like(waveform, dtype=np.complex128)
        for delay, tap, fd in zip(active_delays, taps, path_dopplers):
            delayed = np.zeros_like(waveform, dtype=np.complex128)
            delayed[delay:] = waveform[: waveform.size - delay]
            phase = np.exp(1j * 2.0 * np.pi * fd * n / self.cfg.sample_rate_hz)
            faded += tap * phase * delayed
        n_slots = max(1, int(round(waveform.size / self.cfg.slot_samples)))
        h_grid = time_varying_impulse_response_to_grid(self.cfg, active_delays, taps, path_dopplers, n_slots)
        return faded, h_grid


def impulse_response_to_grid(cfg: NRPhyConfig, h: ComplexArray, n_slots: int = 1) -> ComplexArray:
    """Convert a time-domain channel impulse response to active-subcarrier H_true.

    The simulator channel is block-static over the generated frame, so the same
    frequency response is repeated over every OFDM symbol and slot.
    """
    h = np.asarray(h, dtype=np.complex128)
    h_bins = np.fft.fftshift(np.fft.fft(h, n=cfg.n_fft))
    active_h = h_bins[cfg.active_fft_slice]
    slot_grid = np.tile(active_h[:, None], (1, cfg.symbols_per_slot))
    return np.stack([slot_grid.copy() for _ in range(n_slots)], axis=0)


def mimo_impulse_response_to_grid(cfg: NRPhyConfig, h: ComplexArray, n_slots: int = 1) -> ComplexArray:
    """Convert [N_rx, N_tx, taps] MIMO impulse responses to H grids."""
    h = np.asarray(h, dtype=np.complex128)
    if h.ndim != 3:
        raise ValueError("MIMO impulse response must have shape [N_rx, N_tx, taps].")
    grids = np.zeros(
        (n_slots, h.shape[0], h.shape[1], cfg.n_subcarriers, cfg.symbols_per_slot),
        dtype=np.complex128,
    )
    for rx_idx in range(h.shape[0]):
        for tx_idx in range(h.shape[1]):
            siso_grid = impulse_response_to_grid(cfg, h[rx_idx, tx_idx], n_slots=n_slots)
            grids[:, rx_idx, tx_idx] = siso_grid
    return grids


def time_varying_impulse_response_to_grid(
    cfg: NRPhyConfig,
    delays: np.ndarray,
    taps: ComplexArray,
    path_dopplers_hz: np.ndarray,
    n_slots: int,
) -> ComplexArray:
    """Approximate per-symbol H_true for the lightweight Doppler channel."""
    grids = np.zeros((n_slots, cfg.n_subcarriers, cfg.symbols_per_slot), dtype=np.complex128)
    symbol_offsets = []
    cursor = 0
    for cp in cfg.cp_lengths:
        symbol_offsets.append(cursor + cp + cfg.n_fft / 2.0)
        cursor += cp + cfg.n_fft
    for slot_idx in range(n_slots):
        slot_start = slot_idx * cfg.slot_samples
        for sym_idx, sym_center in enumerate(symbol_offsets):
            sample_time = (slot_start + sym_center) / cfg.sample_rate_hz
            h = np.zeros(int(np.max(delays)) + 1, dtype=np.complex128)
            phases = np.exp(1j * 2.0 * np.pi * path_dopplers_hz * sample_time)
            for delay, tap in zip(delays, taps * phases):
                h[int(delay)] += tap
            h_bins = np.fft.fftshift(np.fft.fft(h, n=cfg.n_fft))
            grids[slot_idx, :, sym_idx] = h_bins[cfg.active_fft_slice]
    return grids
