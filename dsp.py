from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .config import NRPhyConfig
from .resource_grid import ResourceGridMapper
from .utils import ComplexArray, int_array_to_bits


class ChannelEstimator:
    """LS pilot estimation with frequency and time interpolation."""

    def __init__(self, cfg: NRPhyConfig, mapper: ResourceGridMapper) -> None:
        self.cfg = cfg
        self.mapper = mapper

    def estimate_slot(self, rx_grid: ComplexArray, slot_idx: int) -> ComplexArray:
        pilot_mask = self.mapper.pilot_mask_for_slot(slot_idx)
        sparse_h = np.full(rx_grid.shape, np.nan + 1j * np.nan, dtype=np.complex128)

        # Least-squares estimate H(k,l) = Y(k,l) / Xpilot(k,l) at pilot REs.
        for symbol_idx in self.mapper.pilot_symbols:
            subcarriers = np.flatnonzero(pilot_mask[:, symbol_idx])
            pilots = self.mapper.pilot_sequence(slot_idx, symbol_idx, subcarriers.size)
            sparse_h[subcarriers, symbol_idx] = rx_grid[subcarriers, symbol_idx] / pilots

        freq_interp = np.zeros_like(rx_grid)
        all_subcarriers = np.arange(self.cfg.n_subcarriers)
        for symbol_idx in self.mapper.pilot_symbols:
            known_subcarriers = np.flatnonzero(~np.isnan(np.real(sparse_h[:, symbol_idx])))
            known_values = sparse_h[known_subcarriers, symbol_idx]
            real = np.interp(all_subcarriers, known_subcarriers, np.real(known_values))
            imag = np.interp(all_subcarriers, known_subcarriers, np.imag(known_values))
            freq_interp[:, symbol_idx] = real + 1j * imag

        # 2D interpolation: first across frequency in each pilot symbol, then
        # across OFDM-symbol time for every subcarrier. Edges use nearest pilot.
        h_est = np.zeros_like(rx_grid)
        pilot_symbols = np.asarray(self.mapper.pilot_symbols)
        all_symbols = np.arange(self.cfg.symbols_per_slot)
        for subcarrier_idx in range(self.cfg.n_subcarriers):
            known_values = freq_interp[subcarrier_idx, pilot_symbols]
            real = np.interp(all_symbols, pilot_symbols, np.real(known_values))
            imag = np.interp(all_symbols, pilot_symbols, np.imag(known_values))
            h_est[subcarrier_idx, :] = real + 1j * imag
        return h_est

    @staticmethod
    def equalize(
        rx_grid: ComplexArray,
        h_est: ComplexArray,
        method: str = "mmse",
        noise_variance: float = 0.0,
    ) -> ComplexArray:
        eps = 1e-12
        method = method.lower()
        if method == "zf":
            return rx_grid / (h_est + eps)
        if method == "mmse":
            return rx_grid * np.conj(h_est) / (np.abs(h_est) ** 2 + noise_variance + eps)
        raise ValueError("Equalizer method must be zf or mmse.")


@dataclass
class CompressedCSI:
    bitstream: np.ndarray
    metadata: Dict[str, float | int | Tuple[int, ...]]


def compress_csi(
    h_est: ComplexArray,
    subcarrier_stride: int = 4,
    bits_per_component: int = 4,
) -> CompressedCSI:
    """Fallback CSI compression: subcarrier downsampling plus scalar quantization."""
    if bits_per_component < 2 or bits_per_component > 8:
        raise ValueError("bits_per_component should be in [2, 8].")
    if subcarrier_stride <= 0:
        raise ValueError("subcarrier_stride must be positive.")

    h_est = np.asarray(h_est, dtype=np.complex128)
    h_ds = h_est[..., ::subcarrier_stride, :]
    components = np.concatenate([np.real(h_ds).reshape(-1), np.imag(h_ds).reshape(-1)])
    scale = float(np.percentile(np.abs(components), 99.0))
    scale = max(scale, 1e-6)

    levels = (1 << bits_per_component) - 1
    normalized = np.clip((components / scale + 1.0) * 0.5, 0.0, 1.0)
    quantized = np.rint(normalized * levels).astype(np.uint8)
    return CompressedCSI(
        bitstream=int_array_to_bits(quantized, bits_per_component),
        metadata={
            "original_shape": tuple(int(x) for x in h_est.shape),
            "compressed_shape": tuple(int(x) for x in h_ds.shape),
            "subcarrier_stride": int(subcarrier_stride),
            "bits_per_component": int(bits_per_component),
            "scale": scale,
        },
    )


def bits_to_int_array(bits: np.ndarray, bits_per_value: int, count: int) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    needed = count * bits_per_value
    if bits.size < needed:
        bits = np.pad(bits, (0, needed - bits.size), constant_values=0)
    bits = bits[:needed].reshape(-1, bits_per_value)
    weights = (1 << np.arange(bits_per_value - 1, -1, -1, dtype=np.uint32)).reshape(1, -1)
    return np.sum(bits.astype(np.uint32) * weights, axis=1).astype(np.uint8)


def decompress_csi(bitstream: np.ndarray, metadata: Dict[str, object]) -> ComplexArray:
    """Reconstruct a full-size CSI matrix from the fallback compressed bitstream."""
    original_shape = tuple(int(x) for x in metadata["original_shape"])
    compressed_shape = tuple(int(x) for x in metadata["compressed_shape"])
    stride = int(metadata["subcarrier_stride"])
    bits_per_component = int(metadata["bits_per_component"])
    scale = float(metadata["scale"])

    component_count = int(np.prod(compressed_shape))
    quantized = bits_to_int_array(bitstream, bits_per_component, 2 * component_count)
    levels = (1 << bits_per_component) - 1
    values = ((quantized.astype(np.float64) / levels) * 2.0 - 1.0) * scale
    real = values[:component_count].reshape(compressed_shape)
    imag = values[component_count:].reshape(compressed_shape)
    h_ds = real + 1j * imag

    h_full = np.zeros(original_shape, dtype=np.complex128)
    src_positions = np.arange(0, original_shape[-2], stride)[: compressed_shape[-2]]
    all_subcarriers = np.arange(original_shape[-2])
    for slot_idx in range(original_shape[0]):
        for symbol_idx in range(original_shape[-1]):
            known = h_ds[slot_idx, :, symbol_idx]
            if known.size == 1:
                h_full[slot_idx, :, symbol_idx] = known[0]
            else:
                h_full[slot_idx, :, symbol_idx] = np.interp(
                    all_subcarriers, src_positions, np.real(known)
                ) + 1j * np.interp(all_subcarriers, src_positions, np.imag(known))
    return h_full


def csi_nmse(reference: ComplexArray, estimate: ComplexArray) -> float:
    reference = np.asarray(reference, dtype=np.complex128)
    estimate = np.asarray(estimate, dtype=np.complex128)
    numerator = float(np.mean(np.abs(reference - estimate) ** 2))
    denominator = float(np.mean(np.abs(reference) ** 2))
    return numerator / max(denominator, 1e-30)
