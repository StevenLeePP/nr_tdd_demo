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
        rx_grid = np.asarray(rx_grid, dtype=np.complex128)
        if rx_grid.ndim == 3:
            return self.estimate_slot_mimo(rx_grid, slot_idx)
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

    def estimate_slot_mimo(self, rx_grid: ComplexArray, slot_idx: int) -> ComplexArray:
        """Estimate [N_rx, N_tx, subcarrier, symbol] CSI from orthogonal Tx pilots."""
        rx_grid = np.asarray(rx_grid, dtype=np.complex128)
        if rx_grid.shape != (self.cfg.num_rx_antennas, self.cfg.n_subcarriers, self.cfg.symbols_per_slot):
            raise ValueError("MIMO rx_grid must have shape [num_rx, n_subcarriers, symbols].")
        h_est = np.zeros(
            (self.cfg.num_rx_antennas, self.cfg.num_tx_antennas, self.cfg.n_subcarriers, self.cfg.symbols_per_slot),
            dtype=np.complex128,
        )
        all_subcarriers = np.arange(self.cfg.n_subcarriers)
        all_symbols = np.arange(self.cfg.symbols_per_slot)
        pilot_symbols = np.asarray(self.mapper.pilot_symbols)
        for rx_idx in range(self.cfg.num_rx_antennas):
            for tx_idx in range(self.cfg.num_tx_antennas):
                sparse_h = np.full(
                    (self.cfg.n_subcarriers, self.cfg.symbols_per_slot),
                    np.nan + 1j * np.nan,
                    dtype=np.complex128,
                )
                pilot_mask = self.mapper.pilot_mask_for_slot(slot_idx, tx_idx=tx_idx)
                for symbol_idx in self.mapper.pilot_symbols:
                    subcarriers = np.flatnonzero(pilot_mask[:, symbol_idx])
                    pilots = self.mapper.pilot_sequence(slot_idx, symbol_idx, subcarriers.size, tx_idx=tx_idx)
                    sparse_h[subcarriers, symbol_idx] = rx_grid[rx_idx, subcarriers, symbol_idx] / pilots
                freq_interp = np.zeros((self.cfg.n_subcarriers, self.cfg.symbols_per_slot), dtype=np.complex128)
                for symbol_idx in self.mapper.pilot_symbols:
                    known_subcarriers = np.flatnonzero(~np.isnan(np.real(sparse_h[:, symbol_idx])))
                    known_values = sparse_h[known_subcarriers, symbol_idx]
                    real = np.interp(all_subcarriers, known_subcarriers, np.real(known_values))
                    imag = np.interp(all_subcarriers, known_subcarriers, np.imag(known_values))
                    freq_interp[:, symbol_idx] = real + 1j * imag
                for subcarrier_idx in range(self.cfg.n_subcarriers):
                    known_values = freq_interp[subcarrier_idx, pilot_symbols]
                    real = np.interp(all_symbols, pilot_symbols, np.real(known_values))
                    imag = np.interp(all_symbols, pilot_symbols, np.imag(known_values))
                    h_est[rx_idx, tx_idx, subcarrier_idx, :] = real + 1j * imag
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

    @staticmethod
    def equalize_mimo_single_stream(
        rx_grid: ComplexArray,
        h_est: ComplexArray,
        method: str = "mmse",
        noise_variance: float = 0.0,
    ) -> ComplexArray:
        rx_grid = np.asarray(rx_grid, dtype=np.complex128)
        h_est = np.asarray(h_est, dtype=np.complex128)
        if h_est.ndim != 4:
            raise ValueError("h_est must have shape [N_rx, N_tx, subcarrier, symbol].")
        tx_weights = np.full(h_est.shape[1], 1.0 / np.sqrt(h_est.shape[1]), dtype=np.complex128)
        h_eff = np.einsum("rtfs,t->rfs", h_est, tx_weights)
        numerator = np.sum(np.conj(h_eff) * rx_grid, axis=0)
        denom = np.sum(np.abs(h_eff) ** 2, axis=0)
        if method.lower() == "zf":
            return numerator / (denom + 1e-12)
        if method.lower() == "mmse":
            return numerator / (denom + noise_variance + 1e-12)
        raise ValueError("Equalizer method must be zf or mmse.")


def delay_domain_denoise_csi(
    h_est: ComplexArray,
    cfg: NRPhyConfig,
    n_taps: int | None = None,
    time_average: bool = False,
) -> ComplexArray:
    """Apply communication-structured CSI denoising.

    For block-static TDD frames, time averaging across OFDM symbols removes LS
    pilot noise without changing the underlying channel. Optional delay-domain
    truncation can then impose sparse TDL support when the active-band window is
    wide enough for that prior to be helpful.
    """
    if n_taps is None:
        n_taps = cfg.n_fft
    if n_taps <= 0 or n_taps > cfg.n_fft:
        raise ValueError("n_taps must be in [1, n_fft].")
    h_est = np.asarray(h_est, dtype=np.complex128)
    if h_est.ndim == 4:
        return np.stack(
            [
                [
                    delay_domain_denoise_csi(
                        h_est[rx_idx, tx_idx],
                        cfg,
                        n_taps=n_taps,
                        time_average=time_average,
                    )
                    for tx_idx in range(h_est.shape[1])
                ]
                for rx_idx in range(h_est.shape[0])
            ],
            axis=0,
        )
    if h_est.ndim == 5:
        return np.stack(
            [
                delay_domain_denoise_csi(
                    h_est[slot_idx],
                    cfg,
                    n_taps=n_taps,
                    time_average=time_average,
                )
                for slot_idx in range(h_est.shape[0])
            ],
            axis=0,
        )
    if h_est.ndim == 2:
        h_est = h_est[None]
        squeeze = True
    elif h_est.ndim == 3:
        squeeze = False
    else:
        raise ValueError("h_est must have shape [subcarrier, symbol] or [slot, subcarrier, symbol].")
    if time_average:
        mean_h = np.mean(h_est, axis=(0, 2), keepdims=True)
        h_est = np.tile(mean_h, (h_est.shape[0], 1, h_est.shape[2]))
    if n_taps >= cfg.n_fft:
        return h_est[0] if squeeze else h_est

    denoised = np.zeros_like(h_est)
    for slot_idx in range(h_est.shape[0]):
        for symbol_idx in range(h_est.shape[-1]):
            spectrum = np.zeros(cfg.n_fft, dtype=np.complex128)
            spectrum[cfg.active_fft_slice] = h_est[slot_idx, :, symbol_idx]
            impulse = np.fft.ifft(np.fft.ifftshift(spectrum))
            sparse_impulse = np.zeros_like(impulse)
            sparse_impulse[:n_taps] = impulse[:n_taps]
            restored = np.fft.fftshift(np.fft.fft(sparse_impulse))
            denoised[slot_idx, :, symbol_idx] = restored[cfg.active_fft_slice]
    return denoised[0] if squeeze else denoised


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


def compress_csi_delay_domain(
    h_est: ComplexArray,
    cfg: NRPhyConfig,
    n_taps: int = 16,
    bits_per_component: int = 8,
    time_segments: int = 1,
) -> CompressedCSI:
    """Sparse delay-domain CSI compression.

    The NR-like simulator uses short TDL channels, so most channel energy is
    concentrated in the first few delay taps. Feedbacking these taps is far
    more efficient than sending a coarsely downsampled frequency grid.
    """
    if n_taps <= 0 or n_taps > cfg.n_fft:
        raise ValueError("n_taps must be in [1, n_fft].")
    if bits_per_component < 2 or bits_per_component > 8:
        raise ValueError("bits_per_component should be in [2, 8].")
    h_est = np.asarray(h_est, dtype=np.complex128)
    if h_est.ndim != 3:
        raise ValueError("h_est must have shape [slot, subcarrier, symbol].")

    n_slots, n_subcarriers, n_symbols = h_est.shape
    if n_subcarriers != cfg.n_subcarriers:
        raise ValueError("h_est subcarrier dimension must match cfg.n_subcarriers.")
    total_symbols = n_slots * n_symbols
    time_segments = int(np.clip(time_segments, 1, total_symbols))

    flattened = np.transpose(h_est, (0, 2, 1)).reshape(total_symbols, n_subcarriers)
    segment_edges = np.linspace(0, total_symbols, time_segments + 1, dtype=int)
    taps = np.zeros((time_segments, n_taps), dtype=np.complex128)
    for segment_idx in range(time_segments):
        start, end = segment_edges[segment_idx], segment_edges[segment_idx + 1]
        segment_h = np.mean(flattened[start:end], axis=0)
        spectrum = np.zeros(cfg.n_fft, dtype=np.complex128)
        spectrum[cfg.active_fft_slice] = segment_h
        impulse = np.fft.ifft(np.fft.ifftshift(spectrum))
        taps[segment_idx] = impulse[:n_taps]

    components = np.concatenate([np.real(taps).reshape(-1), np.imag(taps).reshape(-1)])
    scale = float(np.percentile(np.abs(components), 99.5))
    scale = max(scale, 1e-8)
    levels = (1 << bits_per_component) - 1
    normalized = np.clip((components / scale + 1.0) * 0.5, 0.0, 1.0)
    quantized = np.rint(normalized * levels).astype(np.uint8)
    return CompressedCSI(
        bitstream=int_array_to_bits(quantized, bits_per_component),
        metadata={
            "method": "delay_domain",
            "original_shape": tuple(int(x) for x in h_est.shape),
            "n_fft": int(cfg.n_fft),
            "active_start": int(cfg.active_fft_slice.start),
            "active_stop": int(cfg.active_fft_slice.stop),
            "n_taps": int(n_taps),
            "time_segments": int(time_segments),
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
    if metadata.get("method") == "delay_domain":
        return decompress_csi_delay_domain(bitstream, metadata)

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
    leading_shape = original_shape[:-2]
    for leading_idx in np.ndindex(leading_shape):
        for symbol_idx in range(original_shape[-1]):
            known = h_ds[leading_idx + (slice(None), symbol_idx)]
            if known.size == 1:
                h_full[leading_idx + (slice(None), symbol_idx)] = known[0]
            else:
                h_full[leading_idx + (slice(None), symbol_idx)] = np.interp(
                    all_subcarriers, src_positions, np.real(known)
                ) + 1j * np.interp(all_subcarriers, src_positions, np.imag(known))
    return h_full


def decompress_csi_delay_domain(bitstream: np.ndarray, metadata: Dict[str, object]) -> ComplexArray:
    original_shape = tuple(int(x) for x in metadata["original_shape"])
    n_fft = int(metadata["n_fft"])
    active_start = int(metadata["active_start"])
    active_stop = int(metadata["active_stop"])
    n_taps = int(metadata["n_taps"])
    time_segments = int(metadata["time_segments"])
    bits_per_component = int(metadata["bits_per_component"])
    scale = float(metadata["scale"])

    component_count = time_segments * n_taps
    quantized = bits_to_int_array(bitstream, bits_per_component, 2 * component_count)
    levels = (1 << bits_per_component) - 1
    values = ((quantized.astype(np.float64) / levels) * 2.0 - 1.0) * scale
    real = values[:component_count].reshape(time_segments, n_taps)
    imag = values[component_count:].reshape(time_segments, n_taps)
    taps = real + 1j * imag

    n_slots, n_subcarriers, n_symbols = original_shape
    total_symbols = n_slots * n_symbols
    segment_edges = np.linspace(0, total_symbols, time_segments + 1, dtype=int)
    flattened = np.zeros((total_symbols, n_subcarriers), dtype=np.complex128)
    for segment_idx in range(time_segments):
        impulse = np.zeros(n_fft, dtype=np.complex128)
        impulse[:n_taps] = taps[segment_idx]
        spectrum = np.fft.fftshift(np.fft.fft(impulse))
        active_h = spectrum[active_start:active_stop]
        start, end = segment_edges[segment_idx], segment_edges[segment_idx + 1]
        flattened[start:end] = active_h
    return flattened.reshape(n_slots, n_symbols, n_subcarriers).transpose(0, 2, 1)


def csi_nmse(reference: ComplexArray, estimate: ComplexArray) -> float:
    reference = np.asarray(reference, dtype=np.complex128)
    estimate = np.asarray(estimate, dtype=np.complex128)
    numerator = float(np.mean(np.abs(reference - estimate) ** 2))
    denominator = float(np.mean(np.abs(reference) ** 2))
    return numerator / max(denominator, 1e-30)
