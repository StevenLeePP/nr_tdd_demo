from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PACKAGE_DIR / "outputs"
DEFAULT_IMAGE_PATH = "/root/lap/semantic/SwinJSCC/Kodak_dataset/kodim01.png"
DEFAULT_SWINJSCC_MODEL_PATH = (
    "/root/lap/semantic/SwinJSCC/models/"
    "SwinJSCC_w_SA_Rayleigh_HRimage_snr_psnr_C96.model"
)


@dataclass(frozen=True)
class NRPhyConfig:
    """Configurable small-scale NR-like numerology for SDR-style simulation."""

    scs_khz: int = 15
    n_fft: int = 1024
    n_subcarriers: int = 600
    n_dl_slots: int = 72
    n_ul_slots: int = 2
    symbols_per_slot: int = 14
    cell_id: int = 42
    snr_db: float = 20.0
    rng_seed: int = 7

    def __post_init__(self) -> None:
        if self.symbols_per_slot != 14:
            raise ValueError("Normal CP NR slots must contain 14 OFDM symbols.")
        if self.n_subcarriers > self.n_fft:
            raise ValueError("n_subcarriers must not exceed n_fft.")
        if self.n_subcarriers <= 0 or self.n_fft <= 0:
            raise ValueError("n_subcarriers and n_fft must be positive.")
        if self.n_subcarriers % 2:
            raise ValueError("n_subcarriers should be even for centered mapping.")
        if self.scs_khz not in {15, 30, 60, 120, 240}:
            raise ValueError("Use an NR SCS value: 15, 30, 60, 120, or 240 kHz.")

    @property
    def subcarrier_spacing_hz(self) -> float:
        return float(self.scs_khz) * 1e3

    @property
    def sample_rate_hz(self) -> float:
        return self.n_fft * self.subcarrier_spacing_hz

    @property
    def cp_lengths(self) -> List[int]:
        """Normal CP samples scaled from the 2048-point 15 kHz reference."""
        short_cp = max(1, int(round(144 * self.n_fft / 2048)))
        long_cp = max(short_cp + 1, int(round(160 * self.n_fft / 2048)))
        return [
            long_cp if symbol_idx in {0, 7} else short_cp
            for symbol_idx in range(self.symbols_per_slot)
        ]

    @property
    def slot_samples(self) -> int:
        return sum(self.n_fft + cp for cp in self.cp_lengths)

    @property
    def active_fft_slice(self) -> slice:
        start = (self.n_fft - self.n_subcarriers) // 2
        return slice(start, start + self.n_subcarriers)


@dataclass(frozen=True)
class SemanticConfig:
    use_real_swinjscc: bool = True
    model_path: str = DEFAULT_SWINJSCC_MODEL_PATH
    model_name: str = "SwinJSCC_w/_SA"
    image_size: int = 256
    rate: int = 96
    model_size: str = "base"


@dataclass(frozen=True)
class ConventionalConfig:
    image_size: int = 256
    h264_crf: int = 28
    ldpc_data_bits: int = 512
    ldpc_parity_bits: int = 512
    ldpc_row_weight: int = 6
    ldpc_max_iters: int = 25
    repetition_factor: int = 3


@dataclass(frozen=True)
class ChannelConfig:
    channel_type: str = "rayleigh"
    delays: tuple[int, ...] = (0, 2, 5)
    powers_db: tuple[float, ...] = (0.0, -3.0, -8.0)
    rician_k_db: float = 6.0


@dataclass(frozen=True)
class DemoConfig:
    image_path: str = DEFAULT_IMAGE_PATH
    output_dir: Path = DEFAULT_OUTPUT_DIR

    def filename_prefix(self, channel_type: str, snr_db: float) -> str:
        snr_tag = f"{snr_db:g}dB".replace(".", "p")
        return f"{channel_type.lower()}_snr_{snr_tag}"
