from __future__ import annotations

import math

import numpy as np


ComplexArray = np.ndarray


def db_to_linear(db_value: float | np.ndarray) -> float | np.ndarray:
    return 10.0 ** (db_value / 10.0)


def qpsk_modulate(bits: np.ndarray) -> ComplexArray:
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if bits.size % 2:
        bits = np.append(bits, 0)
    pairs = bits.reshape(-1, 2)
    real = 1.0 - 2.0 * pairs[:, 0]
    imag = 1.0 - 2.0 * pairs[:, 1]
    return (real + 1j * imag) / math.sqrt(2.0)


def qpsk_demodulate(symbols: ComplexArray) -> np.ndarray:
    bits = np.empty(2 * symbols.size, dtype=np.uint8)
    bits[0::2] = np.real(symbols) < 0.0
    bits[1::2] = np.imag(symbols) < 0.0
    return bits


def int_array_to_bits(values: np.ndarray, bits_per_value: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.uint32).reshape(-1)
    shifts = np.arange(bits_per_value - 1, -1, -1, dtype=np.uint32)
    return ((values[:, None] >> shifts) & 1).astype(np.uint8).reshape(-1)


def generate_gold_bits(length: int, c_init: int) -> np.ndarray:
    """NR-style pseudo-random sequence c(n) based on two length-31 LFSRs."""
    nc = 1600
    size = length + nc + 31
    x1 = np.zeros(size, dtype=np.uint8)
    x2 = np.zeros(size, dtype=np.uint8)
    x1[0] = 1

    c_init = int(c_init) & ((1 << 31) - 1)
    for i in range(31):
        x2[i] = (c_init >> i) & 1

    for n in range(size - 31):
        x1[n + 31] = x1[n + 3] ^ x1[n]
        x2[n + 31] = x2[n + 3] ^ x2[n + 2] ^ x2[n + 1] ^ x2[n]

    return x1[nc : nc + length] ^ x2[nc : nc + length]


def generate_gold_qpsk(num_symbols: int, c_init: int) -> ComplexArray:
    return qpsk_modulate(generate_gold_bits(2 * num_symbols, c_init))

