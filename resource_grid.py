from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from .config import NRPhyConfig
from .utils import ComplexArray, generate_gold_qpsk, qpsk_modulate


class ResourceGridMapper:
    """Maps pilots and data onto a subcarrier x OFDM-symbol resource grid."""

    pilot_symbols = tuple(range(14))
    semantic_label = 1
    conventional_label = 2
    pilot_label = 3

    def __init__(self, cfg: NRPhyConfig, link_name: str) -> None:
        self.cfg = cfg
        self.link_name = link_name

    @property
    def is_downlink(self) -> bool:
        return self.link_name.upper() == "DL"

    @property
    def num_tx_antennas(self) -> int:
        return self.cfg.num_tx_antennas if self.is_downlink else self.cfg.ul_num_tx_antennas

    @property
    def num_rx_antennas(self) -> int:
        return self.cfg.num_rx_antennas if self.is_downlink else self.cfg.ul_num_rx_antennas

    @property
    def mimo_link(self) -> bool:
        return self.num_tx_antennas > 1 or self.num_rx_antennas > 1

    def pilot_mask_for_slot(self, slot_idx: int, tx_idx: int | None = None) -> np.ndarray:
        mask = np.zeros((self.cfg.n_subcarriers, self.cfg.symbols_per_slot), dtype=bool)
        subcarrier_idx = np.arange(self.cfg.n_subcarriers)
        for symbol_idx in self.pilot_symbols:
            # Comb structure in every OFDM symbol. The default spacing is 4,
            # which gives the requested data:pilot RE ratio of 3:1.
            spacing = self.cfg.pilot_spacing * self.num_tx_antennas if self.mimo_link else self.cfg.pilot_spacing
            if self.mimo_link and tx_idx is None:
                for antenna_idx in range(self.num_tx_antennas):
                    shift = (symbol_idx + slot_idx + antenna_idx) % spacing
                    mask[:, symbol_idx] |= ((subcarrier_idx + shift) % spacing) == 0
                continue
            antenna_shift = int(tx_idx or 0)
            shift = (symbol_idx + slot_idx + antenna_shift) % spacing
            mask[:, symbol_idx] = ((subcarrier_idx + shift) % spacing) == 0
        return mask

    def data_mask_for_slot(self, slot_idx: int) -> np.ndarray:
        return ~self.pilot_mask_for_slot(slot_idx)

    def pilot_sequence(self, slot_idx: int, symbol_idx: int, count: int, tx_idx: int = 0) -> ComplexArray:
        link_offset = 0 if self.link_name.upper() == "DL" else 1_000_003
        c_init = self.cfg.cell_id + link_offset + 131 * slot_idx + 17 * symbol_idx + 997 * tx_idx + 2**10
        return generate_gold_qpsk(count, c_init)

    def insert_pilots(self, grid: ComplexArray, slot_idx: int) -> Dict[int, ComplexArray]:
        pilot_sequences: Dict[int, ComplexArray] = {}
        if self.mimo_link:
            if grid.shape != (self.num_tx_antennas, self.cfg.n_subcarriers, self.cfg.symbols_per_slot):
                raise ValueError("MIMO grid must have shape [num_tx, n_subcarriers, symbols].")
            for tx_idx in range(self.num_tx_antennas):
                pilot_mask = self.pilot_mask_for_slot(slot_idx, tx_idx=tx_idx)
                for symbol_idx in self.pilot_symbols:
                    subcarriers = np.flatnonzero(pilot_mask[:, symbol_idx])
                    seq = self.pilot_sequence(slot_idx, symbol_idx, subcarriers.size, tx_idx=tx_idx)
                    grid[tx_idx, subcarriers, symbol_idx] = seq
                    pilot_sequences[(tx_idx, symbol_idx)] = seq
            return pilot_sequences
        pilot_mask = self.pilot_mask_for_slot(slot_idx)
        for symbol_idx in self.pilot_symbols:
            subcarriers = np.flatnonzero(pilot_mask[:, symbol_idx])
            seq = self.pilot_sequence(slot_idx, symbol_idx, subcarriers.size)
            grid[subcarriers, symbol_idx] = seq
            pilot_sequences[symbol_idx] = seq
        return pilot_sequences

    def data_positions(self, slot_idx: int) -> List[Tuple[int, int]]:
        mask = self.data_mask_for_slot(slot_idx)
        return [
            (subcarrier_idx, symbol_idx)
            for symbol_idx in range(self.cfg.symbols_per_slot)
            for subcarrier_idx in range(self.cfg.n_subcarriers)
            if mask[subcarrier_idx, symbol_idx]
        ]

    def data_re_per_slot(self, slot_idx: int = 0) -> int:
        layers = self.num_tx_antennas if (not self.is_downlink and self.mimo_link) else 1
        return len(self.data_positions(slot_idx)) * layers

    def map_symbols_to_slots(
        self, data_symbols: ComplexArray, n_slots: int
    ) -> Tuple[List[ComplexArray], int, List[np.ndarray]]:
        """Map data frequency-first into every non-pilot RE of each slot."""
        data_symbols = np.asarray(data_symbols, dtype=np.complex128).reshape(-1)
        cursor = 0
        grids: List[ComplexArray] = []
        allocations: List[np.ndarray] = []

        for slot_idx in range(n_slots):
            grid_shape = (
                (self.num_tx_antennas, self.cfg.n_subcarriers, self.cfg.symbols_per_slot)
                if self.mimo_link
                else (self.cfg.n_subcarriers, self.cfg.symbols_per_slot)
            )
            grid = np.zeros(grid_shape, dtype=np.complex128)
            allocation = np.zeros(grid.shape, dtype=np.uint8)
            self.insert_pilots(grid, slot_idx)
            if self.mimo_link:
                allocation = np.zeros((self.cfg.n_subcarriers, self.cfg.symbols_per_slot), dtype=np.uint8)
            allocation[self.pilot_mask_for_slot(slot_idx)] = self.pilot_label
            positions = self.data_positions(slot_idx)
            capacity = len(positions) * (self.num_tx_antennas if (self.mimo_link and not self.is_downlink) else 1)
            take = min(capacity, max(0, data_symbols.size - cursor))
            if take:
                if self.mimo_link and not self.is_downlink:
                    local_cursor = 0
                    for subcarrier_idx, symbol_idx in positions:
                        for tx_idx in range(self.num_tx_antennas):
                            if local_cursor >= take:
                                break
                            grid[tx_idx, subcarrier_idx, symbol_idx] = data_symbols[cursor + local_cursor]
                            allocation[subcarrier_idx, symbol_idx] = self.semantic_label
                            local_cursor += 1
                        if local_cursor >= take:
                            break
                else:
                    for (subcarrier_idx, symbol_idx), value in zip(
                        positions[:take], data_symbols[cursor : cursor + take]
                    ):
                        if self.mimo_link:
                            grid[:, subcarrier_idx, symbol_idx] = value / np.sqrt(self.num_tx_antennas)
                        else:
                            grid[subcarrier_idx, symbol_idx] = value
                        allocation[subcarrier_idx, symbol_idx] = self.semantic_label
            cursor += take
            grids.append(grid)
            allocations.append(allocation)

        return grids, cursor, allocations

    def map_dual_streams_to_slots(
        self,
        semantic_symbols: ComplexArray,
        conventional_symbols: ComplexArray,
        n_slots: int,
    ) -> Tuple[List[ComplexArray], int, int, List[np.ndarray]]:
        """Map semantic symbols first, then H264+LDPC symbols, into data REs."""
        semantic_symbols = np.asarray(semantic_symbols, dtype=np.complex128).reshape(-1)
        conventional_symbols = np.asarray(conventional_symbols, dtype=np.complex128).reshape(-1)
        total_capacity = sum(self.data_re_per_slot(slot_idx) for slot_idx in range(n_slots))
        if semantic_symbols.size + conventional_symbols.size > total_capacity:
            raise ValueError(
                "DL grid cannot fit both streams: "
                f"semantic={semantic_symbols.size}, conventional={conventional_symbols.size}, "
                f"capacity={total_capacity}."
            )

        grids: List[ComplexArray] = []
        allocations: List[np.ndarray] = []
        semantic_cursor = 0
        conventional_cursor = 0

        for slot_idx in range(n_slots):
            grid_shape = (
                (self.num_tx_antennas, self.cfg.n_subcarriers, self.cfg.symbols_per_slot)
                if self.mimo_link
                else (self.cfg.n_subcarriers, self.cfg.symbols_per_slot)
            )
            grid = np.zeros(grid_shape, dtype=np.complex128)
            allocation = np.zeros(grid.shape, dtype=np.uint8)
            self.insert_pilots(grid, slot_idx)
            if self.mimo_link:
                allocation = np.zeros((self.cfg.n_subcarriers, self.cfg.symbols_per_slot), dtype=np.uint8)
            allocation[self.pilot_mask_for_slot(slot_idx)] = self.pilot_label

            for subcarrier_idx, symbol_idx in self.data_positions(slot_idx):
                if semantic_cursor < semantic_symbols.size:
                    if self.mimo_link:
                        grid[:, subcarrier_idx, symbol_idx] = semantic_symbols[semantic_cursor] / np.sqrt(
                            self.num_tx_antennas
                        )
                    else:
                        grid[subcarrier_idx, symbol_idx] = semantic_symbols[semantic_cursor]
                    allocation[subcarrier_idx, symbol_idx] = self.semantic_label
                    semantic_cursor += 1
                elif conventional_cursor < conventional_symbols.size:
                    if self.mimo_link:
                        grid[:, subcarrier_idx, symbol_idx] = conventional_symbols[conventional_cursor] / np.sqrt(
                            self.num_tx_antennas
                        )
                    else:
                        grid[subcarrier_idx, symbol_idx] = conventional_symbols[conventional_cursor]
                    allocation[subcarrier_idx, symbol_idx] = self.conventional_label
                    conventional_cursor += 1

            grids.append(grid)
            allocations.append(allocation)

        return grids, semantic_cursor, conventional_cursor, allocations

    def map_bits_to_slots(self, bits: np.ndarray, n_slots: int) -> Tuple[List[ComplexArray], int, List[np.ndarray]]:
        return self.map_symbols_to_slots(qpsk_modulate(bits), n_slots)

    def extract_data_symbols(self, grids: Sequence[ComplexArray]) -> ComplexArray:
        values: List[complex] = []
        for slot_idx, grid in enumerate(grids):
            for subcarrier_idx, symbol_idx in self.data_positions(slot_idx):
                if np.asarray(grid).ndim == 3:
                    values.extend(grid[:, subcarrier_idx, symbol_idx])
                else:
                    values.append(grid[subcarrier_idx, symbol_idx])
        return np.asarray(values, dtype=np.complex128)

    def extract_allocated_symbols(
        self,
        grids: Sequence[ComplexArray],
        allocations: Sequence[np.ndarray],
        label: int,
    ) -> ComplexArray:
        values: List[complex] = []
        for grid, allocation in zip(grids, allocations):
            grid = grid[0] if np.asarray(grid).ndim == 3 else grid
            for symbol_idx in range(self.cfg.symbols_per_slot):
                subcarriers = np.flatnonzero(allocation[:, symbol_idx] == label)
                values.extend(grid[subcarriers, symbol_idx])
        return np.asarray(values, dtype=np.complex128)
