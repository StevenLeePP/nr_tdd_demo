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

    def pilot_mask_for_slot(self, slot_idx: int) -> np.ndarray:
        mask = np.zeros((self.cfg.n_subcarriers, self.cfg.symbols_per_slot), dtype=bool)
        subcarrier_idx = np.arange(self.cfg.n_subcarriers)
        for symbol_idx in self.pilot_symbols:
            # Comb structure in every OFDM symbol. The default spacing is 4,
            # which gives the requested data:pilot RE ratio of 3:1.
            spacing = self.cfg.pilot_spacing
            shift = (symbol_idx + slot_idx) % spacing
            mask[:, symbol_idx] = ((subcarrier_idx + shift) % spacing) == 0
        return mask

    def data_mask_for_slot(self, slot_idx: int) -> np.ndarray:
        return ~self.pilot_mask_for_slot(slot_idx)

    def pilot_sequence(self, slot_idx: int, symbol_idx: int, count: int) -> ComplexArray:
        link_offset = 0 if self.link_name.upper() == "DL" else 1_000_003
        c_init = self.cfg.cell_id + link_offset + 131 * slot_idx + 17 * symbol_idx + 2**10
        return generate_gold_qpsk(count, c_init)

    def insert_pilots(self, grid: ComplexArray, slot_idx: int) -> Dict[int, ComplexArray]:
        pilot_sequences: Dict[int, ComplexArray] = {}
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
        return len(self.data_positions(slot_idx))

    def map_symbols_to_slots(
        self, data_symbols: ComplexArray, n_slots: int
    ) -> Tuple[List[ComplexArray], int, List[np.ndarray]]:
        """Map data frequency-first into every non-pilot RE of each slot."""
        data_symbols = np.asarray(data_symbols, dtype=np.complex128).reshape(-1)
        cursor = 0
        grids: List[ComplexArray] = []
        allocations: List[np.ndarray] = []

        for slot_idx in range(n_slots):
            grid = np.zeros(
                (self.cfg.n_subcarriers, self.cfg.symbols_per_slot),
                dtype=np.complex128,
            )
            allocation = np.zeros(grid.shape, dtype=np.uint8)
            self.insert_pilots(grid, slot_idx)
            allocation[self.pilot_mask_for_slot(slot_idx)] = self.pilot_label
            positions = self.data_positions(slot_idx)
            take = min(len(positions), max(0, data_symbols.size - cursor))
            if take:
                for (subcarrier_idx, symbol_idx), value in zip(
                    positions[:take], data_symbols[cursor : cursor + take]
                ):
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
            grid = np.zeros(
                (self.cfg.n_subcarriers, self.cfg.symbols_per_slot),
                dtype=np.complex128,
            )
            allocation = np.zeros(grid.shape, dtype=np.uint8)
            self.insert_pilots(grid, slot_idx)
            allocation[self.pilot_mask_for_slot(slot_idx)] = self.pilot_label

            for subcarrier_idx, symbol_idx in self.data_positions(slot_idx):
                if semantic_cursor < semantic_symbols.size:
                    grid[subcarrier_idx, symbol_idx] = semantic_symbols[semantic_cursor]
                    allocation[subcarrier_idx, symbol_idx] = self.semantic_label
                    semantic_cursor += 1
                elif conventional_cursor < conventional_symbols.size:
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
            for symbol_idx in range(self.cfg.symbols_per_slot):
                subcarriers = np.flatnonzero(allocation[:, symbol_idx] == label)
                values.extend(grid[subcarriers, symbol_idx])
        return np.asarray(values, dtype=np.complex128)
