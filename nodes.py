from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .config import NRPhyConfig
from .dsp import ChannelEstimator, CompressedCSI
from .ofdm import OFDMModem
from .resource_grid import ResourceGridMapper
from .semantic import SwinJSCCInterface
from .utils import ComplexArray, qpsk_demodulate


@dataclass
class DownlinkBuild:
    waveform: ComplexArray
    semantic_used_symbols: int
    conventional_used_symbols: int
    grids: List[ComplexArray]
    allocations: List[np.ndarray]


@dataclass
class DownlinkReceive:
    reconstructed: object
    semantic_pre_equalized: ComplexArray
    semantic_equalized: ComplexArray
    conventional_pre_equalized: ComplexArray
    conventional_equalized: ComplexArray
    h_est: ComplexArray


class BaseStation:
    def __init__(self, cfg: NRPhyConfig, codec: SwinJSCCInterface) -> None:
        self.cfg = cfg
        self.codec = codec
        self.mapper = ResourceGridMapper(cfg, "DL")
        self.ul_mapper = ResourceGridMapper(cfg, "UL")
        self.modem = OFDMModem(cfg)
        self.ul_estimator = ChannelEstimator(cfg, self.ul_mapper)

    def build_downlink(
        self,
        image_path: Optional[str],
        conventional_symbols: ComplexArray,
    ) -> DownlinkBuild:
        total_re = sum(self.mapper.data_re_per_slot(slot_idx) for slot_idx in range(self.cfg.n_dl_slots))
        conventional_symbols = np.asarray(conventional_symbols, dtype=np.complex128).reshape(-1)
        semantic_symbols = self.codec.encode_image(image_path, total_re - conventional_symbols.size)
        grids, semantic_used, conventional_used, allocations = self.mapper.map_dual_streams_to_slots(
            semantic_symbols,
            conventional_symbols,
            self.cfg.n_dl_slots,
        )
        return DownlinkBuild(
            waveform=self.modem.modulate_frame(grids),
            semantic_used_symbols=semantic_used,
            conventional_used_symbols=conventional_used,
            grids=grids,
            allocations=allocations,
        )

    def receive_csi_feedback(
        self,
        waveform: ComplexArray,
        noise_variance: float,
        expected_bits: int,
    ) -> np.ndarray:
        rx_grids = self.modem.demodulate_frame(waveform, self.cfg.n_ul_slots)
        equalized_grids = []
        for slot_idx, rx_grid in enumerate(rx_grids):
            h_est = self.ul_estimator.estimate_slot(rx_grid, slot_idx)
            equalized_grids.append(
                self.ul_estimator.equalize(rx_grid, h_est, method="mmse", noise_variance=noise_variance)
            )
        feedback_symbols = self.ul_mapper.extract_data_symbols(equalized_grids)
        return qpsk_demodulate(feedback_symbols)[:expected_bits]


class UserEquipment:
    def __init__(self, cfg: NRPhyConfig, codec: SwinJSCCInterface) -> None:
        self.cfg = cfg
        self.codec = codec
        self.mapper = ResourceGridMapper(cfg, "DL")
        self.ul_mapper = ResourceGridMapper(cfg, "UL")
        self.modem = OFDMModem(cfg)
        self.estimator = ChannelEstimator(cfg, self.mapper)

    def receive_downlink(
        self,
        waveform: ComplexArray,
        noise_variance: float,
        allocations: List[np.ndarray],
        output_path: Optional[str] = None,
    ) -> DownlinkReceive:
        rx_grids = self.modem.demodulate_frame(waveform, self.cfg.n_dl_slots)
        h_est_slots = []
        equalized_grids = []

        for slot_idx, rx_grid in enumerate(rx_grids):
            h_est = self.estimator.estimate_slot(rx_grid, slot_idx)
            h_est_slots.append(h_est)
            equalized_grids.append(
                self.estimator.equalize(rx_grid, h_est, method="mmse", noise_variance=noise_variance)
            )

        semantic_pre = self.mapper.extract_allocated_symbols(
            rx_grids, allocations, self.mapper.semantic_label
        )
        semantic_eq = self.mapper.extract_allocated_symbols(
            equalized_grids, allocations, self.mapper.semantic_label
        )
        conventional_pre = self.mapper.extract_allocated_symbols(
            rx_grids, allocations, self.mapper.conventional_label
        )
        conventional_eq = self.mapper.extract_allocated_symbols(
            equalized_grids, allocations, self.mapper.conventional_label
        )
        reconstructed = self.codec.decode_symbols(semantic_eq, output_path=output_path)
        return DownlinkReceive(
            reconstructed=reconstructed,
            semantic_pre_equalized=semantic_pre,
            semantic_equalized=semantic_eq,
            conventional_pre_equalized=conventional_pre,
            conventional_equalized=conventional_eq,
            h_est=np.stack(h_est_slots, axis=0),
        )

    def build_csi_feedback(self, compressed_csi: CompressedCSI) -> Tuple[ComplexArray, int, List[ComplexArray]]:
        ul_grids, used_symbols, _ = self.ul_mapper.map_bits_to_slots(
            compressed_csi.bitstream, self.cfg.n_ul_slots
        )
        return self.modem.modulate_frame(ul_grids), used_symbols, ul_grids
