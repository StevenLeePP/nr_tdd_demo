from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np

from .channel import MultipathChannel, impulse_response_to_grid
from .config import ChannelConfig, ConventionalConfig, DemoConfig, NRPhyConfig, SemanticConfig
from .conventional import H264LDPCImageCodec
from .dsp import CompressedCSI, compress_csi, compress_csi_delay_domain, csi_nmse, decompress_csi
from .learned_estimator import LearnedChannelEstimator
from .nodes import BaseStation, UserEquipment
from .semantic import SwinJSCCInterface
from .utils import ComplexArray, qpsk_demodulate, qpsk_modulate
from .visualization import (
    build_output_paths,
    plot_constellation_comparison,
    plot_frame_structure,
    plot_reconstruction_comparison,
    plot_time_frequency_resource_grid,
)


ESTIMATOR_ALIASES = {
    "comm_foundation": "comm_foundation_trained",
}
ESTIMATOR_MODES = {
    "ls",
    "ls_smoothing",
    "comm_foundation_untrained",
    "comm_foundation_trained",
}


@dataclass
class SimulationResult:
    reconstructed: object
    dl_used_symbols: int
    csi_bits: int
    ul_used_symbols: int
    bs_feedback_bit_errors: int
    csi_metadata: Dict[str, float | int | Tuple[int, ...]]
    dl_channel_h: ComplexArray
    ul_channel_h: ComplexArray
    target_snr_db: float
    dl_measured_snr_db: float
    ul_measured_snr_db: float
    channel_type: str
    channel_doppler_hz: float
    output_paths: Dict[str, str]
    conventional_metadata: Dict[str, int | float | str]
    csi_feedback_quality: Dict[str, int | float]
    channel_estimation_quality: Dict[str, int | float | str]

    def summary(self) -> Dict[str, object]:
        return {
            "channel_type": self.channel_type,
            "target_snr_db": self.target_snr_db,
            "dl_measured_snr_db": self.dl_measured_snr_db,
            "ul_measured_snr_db": self.ul_measured_snr_db,
            "dl_used_symbols": self.dl_used_symbols,
            "channel_doppler_hz": self.channel_doppler_hz,
            "csi_bits": self.csi_bits,
            "ul_used_symbols": self.ul_used_symbols,
            "bs_feedback_bit_errors": self.bs_feedback_bit_errors,
            "csi_metadata": self.csi_metadata,
            "dl_channel_taps": [
                [float(np.real(x)), float(np.imag(x))] for x in self.dl_channel_h
            ],
            "ul_channel_taps": [
                [float(np.real(x)), float(np.imag(x))] for x in self.ul_channel_h
            ],
            "output_paths": self.output_paths,
            "conventional": self.conventional_metadata,
            "csi_feedback_quality": self.csi_feedback_quality,
            "channel_estimation_quality": self.channel_estimation_quality,
            "reconstructed": self.reconstructed,
        }


class TDDPhysicalLayerSimulation:
    """End-to-end BS-to-UE semantic DL and UE-to-BS CSI feedback loop."""

    def __init__(
        self,
        phy_cfg: NRPhyConfig,
        channel_cfg: ChannelConfig,
        semantic_cfg: SemanticConfig,
        conventional_cfg: ConventionalConfig,
        demo_cfg: DemoConfig,
        reciprocal_tdd: bool = True,
        channel_estimator: str = "ls",
        comm_foundation_checkpoint: str | None = None,
    ) -> None:
        self.phy_cfg = phy_cfg
        self.channel_cfg = channel_cfg
        self.semantic_cfg = semantic_cfg
        self.conventional_cfg = conventional_cfg
        self.demo_cfg = demo_cfg
        self.channel_estimator = ESTIMATOR_ALIASES.get(channel_estimator, channel_estimator)
        if self.channel_estimator not in ESTIMATOR_MODES:
            raise ValueError(f"channel_estimator must be one of {sorted(ESTIMATOR_MODES)}.")
        self.output_paths = build_output_paths(
            self.demo_cfg.output_dir,
            self.channel_cfg.channel_type,
            self.phy_cfg.snr_db,
            estimator_tag=self.channel_estimator,
        )

        self.codec = SwinJSCCInterface(
            semantic_cfg,
            snr_db=phy_cfg.snr_db,
            channel_type=channel_cfg.channel_type,
        )
        self.conventional_codec = H264LDPCImageCodec(conventional_cfg)
        learned_estimator = None
        use_time_average = channel_cfg.doppler_hz == 0.0
        if self.channel_estimator == "ls_smoothing":
            learned_estimator = LearnedChannelEstimator(
                None,
                phy_cfg,
                time_average=use_time_average,
                use_neural_model=False,
            )
        elif self.channel_estimator == "comm_foundation_untrained":
            learned_estimator = LearnedChannelEstimator(
                None,
                phy_cfg,
                time_average=use_time_average,
                use_neural_model=True,
            )
        elif self.channel_estimator == "comm_foundation_trained":
            if not comm_foundation_checkpoint:
                raise ValueError("comm_foundation_trained estimator requires a checkpoint path.")
            learned_estimator = LearnedChannelEstimator(
                comm_foundation_checkpoint,
                phy_cfg,
                time_average=use_time_average,
                use_neural_model=True,
            )

        self.bs = BaseStation(phy_cfg, self.codec)
        self.ue = UserEquipment(phy_cfg, self.codec, learned_estimator=learned_estimator)
        rng = np.random.default_rng(phy_cfg.rng_seed)
        self.channel = MultipathChannel(
            phy_cfg,
            channel_type=channel_cfg.channel_type,
            delays=channel_cfg.delays,
            powers_db=channel_cfg.powers_db,
            rician_k_db=channel_cfg.rician_k_db,
            doppler_hz=channel_cfg.doppler_hz,
            rng=rng,
        )
        self.reciprocal_tdd = reciprocal_tdd

    def run(self) -> SimulationResult:
        conventional_tx = self.conventional_codec.encode_image(self.demo_cfg.image_path)
        conventional_symbols = qpsk_modulate(conventional_tx.tx_bits)

        dl_build = self.bs.build_downlink(self.demo_cfg.image_path, conventional_symbols)
        dl_channel = self.channel.transmit(dl_build.waveform)
        h_true = (
            dl_channel.frequency_response_grid
            if dl_channel.frequency_response_grid is not None
            else impulse_response_to_grid(self.phy_cfg, dl_channel.impulse_response, n_slots=self.phy_cfg.n_dl_slots)
        )

        dl_rx = self.ue.receive_downlink(
            dl_channel.waveform,
            noise_variance=dl_channel.noise_variance,
            allocations=dl_build.allocations,
            output_path=self.output_paths["semantic_reconstructed"],
        )

        conventional_rx_bits = qpsk_demodulate(dl_rx.conventional_equalized)[: conventional_tx.tx_bits.size]
        traditional_recon, traditional_meta = self.conventional_codec.decode_bits(
            conventional_rx_bits,
            original_bit_count=conventional_tx.original_bits.size,
            output_path=self.output_paths["traditional_reconstructed"],
        )
        coded_bit_errors = int(np.sum(conventional_rx_bits != conventional_tx.tx_bits[: conventional_rx_bits.size]))
        recovered_ldpc_bits = self.conventional_codec._majority_combine_repetitions(conventional_rx_bits)
        ldpc_input_bit_errors = int(
            np.sum(recovered_ldpc_bits != conventional_tx.ldpc_bits[: recovered_ldpc_bits.size])
        )
        decoded_bit_errors = self._count_decoded_payload_errors(
            conventional_rx_bits,
            conventional_tx.original_bits,
        )
        traditional_meta.update(conventional_tx.metadata)
        traditional_meta.update(
            {
                "coded_bit_errors": coded_bit_errors,
                "ldpc_input_bit_errors_after_repetition": ldpc_input_bit_errors,
                "decoded_payload_bit_errors": decoded_bit_errors,
                "psnr_db": self._psnr(conventional_tx.original_image, traditional_recon),
            }
        )

        compressed = self._compress_csi_for_ul_capacity(dl_rx.h_est)
        ul_tx, ul_used_symbols, ul_grids = self.ue.build_csi_feedback(compressed)
        ul_h = dl_channel.impulse_response if self.reciprocal_tdd else None
        ul_channel = self.channel.transmit(ul_tx, h=ul_h)
        recovered_bits = self.bs.receive_csi_feedback(
            ul_channel.waveform,
            noise_variance=ul_channel.noise_variance,
            expected_bits=compressed.bitstream.size,
        )

        comparable = min(recovered_bits.size, compressed.bitstream.size)
        bit_errors = int(np.sum(recovered_bits[:comparable] != compressed.bitstream[:comparable]))
        bit_errors += int(compressed.bitstream.size - comparable)
        csi_quality = self._evaluate_csi_feedback(dl_rx.h_est, compressed, recovered_bits, bit_errors, h_true=h_true)
        ce_quality = {
            "estimator": self.channel_estimator,
            "h_est_nmse": float(csi_nmse(h_true, dl_rx.h_est)),
            "h_est_nmse_db": float(10.0 * np.log10(max(csi_nmse(h_true, dl_rx.h_est), 1e-30))),
            "semantic_evm": float(self._evm(self.codec.last_tx_symbols, dl_rx.semantic_equalized)),
            "semantic_evm_db": float(
                20.0 * np.log10(max(self._evm(self.codec.last_tx_symbols, dl_rx.semantic_equalized), 1e-30))
            ),
            "estimator_inference_time_ms": float(dl_rx.estimator_inference_time_ms),
        }

        self._write_visualizations(
            dl_grids=dl_build.grids,
            ul_grids=ul_grids,
            dl_allocations=dl_build.allocations,
            dl_rx=dl_rx,
            traditional_reconstruction=traditional_recon,
            traditional_psnr_db=float(traditional_meta["psnr_db"]),
        )

        return SimulationResult(
            reconstructed=dl_rx.reconstructed,
            dl_used_symbols=dl_build.semantic_used_symbols + dl_build.conventional_used_symbols,
            csi_bits=int(compressed.bitstream.size),
            ul_used_symbols=ul_used_symbols,
            bs_feedback_bit_errors=bit_errors,
            csi_metadata=compressed.metadata,
            dl_channel_h=dl_channel.impulse_response,
            ul_channel_h=ul_channel.impulse_response,
            target_snr_db=self.phy_cfg.snr_db,
            dl_measured_snr_db=dl_channel.measured_snr_db,
            ul_measured_snr_db=ul_channel.measured_snr_db,
            channel_type=self.channel_cfg.channel_type,
            channel_doppler_hz=self.channel_cfg.doppler_hz,
            output_paths=self.output_paths,
            conventional_metadata=traditional_meta,
            csi_feedback_quality=csi_quality,
            channel_estimation_quality=ce_quality,
        )

    def _write_visualizations(
        self,
        dl_grids: Sequence[ComplexArray],
        ul_grids: Sequence[ComplexArray],
        dl_allocations: Sequence[np.ndarray],
        dl_rx,
        traditional_reconstruction: np.ndarray,
        traditional_psnr_db: float,
    ) -> None:
        plot_constellation_comparison(
            dl_rx.semantic_pre_equalized,
            dl_rx.semantic_equalized,
            self.output_paths["semantic_constellation"],
            n_complex=self.codec.semantic_state.get("n_complex_symbols"),
            title_prefix="SwinJSCC semantic stream",
        )
        plot_constellation_comparison(
            dl_rx.conventional_pre_equalized,
            dl_rx.conventional_equalized,
            self.output_paths["traditional_constellation"],
            title_prefix="H.264+LDPC stream",
        )
        if self.codec.last_original is not None and self.codec.last_reconstruction is not None:
            plot_reconstruction_comparison(
                self.codec.last_original,
                self.codec.last_reconstruction,
                traditional_reconstruction,
                self.output_paths["reconstruction"],
                semantic_psnr_db=float(self.codec.last_metrics.get("psnr_db", float("nan"))),
                traditional_psnr_db=traditional_psnr_db,
            )
        plot_time_frequency_resource_grid(
            self.phy_cfg,
            dl_grids,
            ul_grids,
            dl_allocations,
            self.output_paths["resource_grid"],
        )
        plot_frame_structure(self.phy_cfg, self.output_paths["frame_structure"])

    def _compress_csi_for_ul_capacity(self, h_est: ComplexArray) -> CompressedCSI:
        """Choose the smallest downsampling stride that fits the configured UL."""
        capacity_bits = 2 * sum(
            self.ue.ul_mapper.data_re_per_slot(slot_idx)
            for slot_idx in range(self.phy_cfg.n_ul_slots)
        )
        bits_per_tap = 2 * 8
        n_taps = min(16, self.phy_cfg.n_fft)
        bits_per_segment = n_taps * bits_per_tap
        max_segments = capacity_bits // bits_per_segment
        total_symbols = h_est.shape[0] * h_est.shape[-1]
        if max_segments >= 1:
            compressed = compress_csi_delay_domain(
                h_est,
                self.phy_cfg,
                n_taps=n_taps,
                bits_per_component=8,
                time_segments=min(total_symbols, int(max_segments)),
            )
            compressed.metadata["ul_capacity_bits"] = int(capacity_bits)
            return compressed
        for stride in range(4, self.phy_cfg.n_subcarriers + 1):
            compressed = compress_csi(h_est, subcarrier_stride=stride)
            if compressed.bitstream.size <= capacity_bits:
                compressed.metadata["ul_capacity_bits"] = int(capacity_bits)
                compressed.metadata["method"] = "frequency_stride"
                return compressed
        raise ValueError(
            "Configured UL grid cannot carry even the most aggressively downsampled CSI."
        )

    def _evaluate_csi_feedback(
        self,
        h_est: ComplexArray,
        compressed: CompressedCSI,
        recovered_bits: np.ndarray,
        bit_errors: int,
        h_true: ComplexArray | None = None,
    ) -> Dict[str, int | float]:
        ue_compressed_h = decompress_csi(compressed.bitstream, compressed.metadata)
        bs_recovered_h = decompress_csi(recovered_bits, compressed.metadata)
        total_bits = int(compressed.bitstream.size)
        quality = {
            "feedback_method": str(compressed.metadata.get("method", "frequency_stride")),
            "feedback_bits": total_bits,
            "bs_received_bits": int(recovered_bits.size),
            "feedback_bit_errors": int(bit_errors),
            "feedback_ber": float(bit_errors / max(total_bits, 1)),
            "ue_compression_nmse": float(csi_nmse(h_est, ue_compressed_h)),
            "ue_compression_nmse_db": float(10.0 * np.log10(max(csi_nmse(h_est, ue_compressed_h), 1e-30))),
            "bs_recovered_csi_nmse": float(csi_nmse(h_est, bs_recovered_h)),
            "bs_recovered_csi_nmse_db": float(10.0 * np.log10(max(csi_nmse(h_est, bs_recovered_h), 1e-30))),
        }
        if h_true is not None:
            quality.update(
                {
                    "ue_compression_true_nmse": float(csi_nmse(h_true, ue_compressed_h)),
                    "ue_compression_true_nmse_db": float(
                        10.0 * np.log10(max(csi_nmse(h_true, ue_compressed_h), 1e-30))
                    ),
                    "bs_recovered_true_csi_nmse": float(csi_nmse(h_true, bs_recovered_h)),
                    "bs_recovered_true_csi_nmse_db": float(
                        10.0 * np.log10(max(csi_nmse(h_true, bs_recovered_h), 1e-30))
                    ),
                }
            )
        return quality

    def _count_decoded_payload_errors(self, received_coded_bits: np.ndarray, payload_bits: np.ndarray) -> int:
        ldpc_input_bits = self.conventional_codec._majority_combine_repetitions(received_coded_bits)
        decoded_bits, _ = self.conventional_codec.ldpc.decode(ldpc_input_bits, payload_bits.size)
        return int(np.sum(decoded_bits[: payload_bits.size] != payload_bits))

    @staticmethod
    def _psnr(reference: np.ndarray, estimate: np.ndarray) -> float:
        mse = float(np.mean((np.asarray(reference) - np.asarray(estimate)) ** 2))
        if mse <= 1e-12:
            return float("inf")
        return 10.0 * np.log10(1.0 / mse)

    @staticmethod
    def _evm(reference: ComplexArray | None, estimate: ComplexArray) -> float:
        if reference is None:
            return float("nan")
        reference = np.asarray(reference, dtype=np.complex128).reshape(-1)
        estimate = np.asarray(estimate, dtype=np.complex128).reshape(-1)[: reference.size]
        return float(
            np.sqrt(
                np.mean(np.abs(estimate - reference[: estimate.size]) ** 2)
                / max(np.mean(np.abs(reference[: estimate.size]) ** 2), 1e-30)
            )
        )
