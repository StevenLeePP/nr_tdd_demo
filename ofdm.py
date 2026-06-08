from __future__ import annotations

import math
from typing import List, Sequence

import numpy as np

from .config import NRPhyConfig
from .utils import ComplexArray


class OFDMModem:
    """OFDM modulation/demodulation with symbol-dependent normal CP lengths."""

    def __init__(self, cfg: NRPhyConfig) -> None:
        self.cfg = cfg

    def modulate_slot(self, grid: ComplexArray) -> ComplexArray:
        if grid.shape != (self.cfg.n_subcarriers, self.cfg.symbols_per_slot):
            raise ValueError("Grid shape must be (n_subcarriers, 14).")

        symbols = []
        for symbol_idx, cp_len in enumerate(self.cfg.cp_lengths):
            fft_bins = np.zeros(self.cfg.n_fft, dtype=np.complex128)
            fft_bins[self.cfg.active_fft_slice] = grid[:, symbol_idx]
            time_symbol = np.fft.ifft(np.fft.ifftshift(fft_bins)) * math.sqrt(self.cfg.n_fft)
            symbols.append(np.concatenate([time_symbol[-cp_len:], time_symbol]))
        return np.concatenate(symbols)

    def demodulate_slot(self, waveform: ComplexArray) -> ComplexArray:
        expected = self.cfg.slot_samples
        if waveform.size < expected:
            raise ValueError(f"Need {expected} samples for one slot, got {waveform.size}.")

        grid = np.zeros((self.cfg.n_subcarriers, self.cfg.symbols_per_slot), dtype=np.complex128)
        cursor = 0
        for symbol_idx, cp_len in enumerate(self.cfg.cp_lengths):
            cursor += cp_len
            useful = waveform[cursor : cursor + self.cfg.n_fft]
            cursor += self.cfg.n_fft
            fft_bins = np.fft.fftshift(np.fft.fft(useful) / math.sqrt(self.cfg.n_fft))
            grid[:, symbol_idx] = fft_bins[self.cfg.active_fft_slice]
        return grid

    def modulate_frame(self, grids: Sequence[ComplexArray]) -> ComplexArray:
        return np.concatenate([self.modulate_slot(grid) for grid in grids])

    def demodulate_frame(self, waveform: ComplexArray, n_slots: int) -> List[ComplexArray]:
        grids = []
        for slot_idx in range(n_slots):
            start = slot_idx * self.cfg.slot_samples
            stop = start + self.cfg.slot_samples
            grids.append(self.demodulate_slot(waveform[start:stop]))
        return grids
