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
        grid = np.asarray(grid, dtype=np.complex128)
        if grid.ndim == 3:
            return np.stack([self.modulate_slot(grid[tx_idx]) for tx_idx in range(grid.shape[0])], axis=0)
        if grid.shape != (self.cfg.n_subcarriers, self.cfg.symbols_per_slot):
            raise ValueError("Grid shape must be (n_subcarriers, 14) or (num_tx, n_subcarriers, 14).")

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
        slots = [self.modulate_slot(grid) for grid in grids]
        axis = 1 if slots and np.asarray(slots[0]).ndim == 2 else 0
        return np.concatenate(slots, axis=axis)

    def demodulate_frame(self, waveform: ComplexArray, n_slots: int) -> List[ComplexArray]:
        waveform = np.asarray(waveform, dtype=np.complex128)
        if waveform.ndim == 2:
            per_rx = [self.demodulate_frame(waveform[rx_idx], n_slots) for rx_idx in range(waveform.shape[0])]
            return [np.stack([per_rx[rx_idx][slot_idx] for rx_idx in range(waveform.shape[0])], axis=0) for slot_idx in range(n_slots)]
        grids = []
        for slot_idx in range(n_slots):
            start = slot_idx * self.cfg.slot_samples
            stop = start + self.cfg.slot_samples
            grids.append(self.demodulate_slot(waveform[start:stop]))
        return grids
