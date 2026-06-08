#!/usr/bin/env python3
"""Export CSI/IQ samples for a communication foundation-model prototype."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PACKAGE_DIR = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = PACKAGE_DIR.parent
sys.path.insert(0, str(PACKAGE_PARENT))

from nr_tdd_semantic.channel import (  # noqa: E402
    MultipathChannel,
    impulse_response_to_grid,
)
from nr_tdd_semantic.config import NRPhyConfig  # noqa: E402
from nr_tdd_semantic.dsp import ChannelEstimator, csi_nmse  # noqa: E402
from nr_tdd_semantic.ofdm import OFDMModem  # noqa: E402
from nr_tdd_semantic.resource_grid import ResourceGridMapper  # noqa: E402
from nr_tdd_semantic.utils import qpsk_modulate  # noqa: E402


def choose_channel_mode(rng: np.random.Generator, mode: str) -> str:
    if mode == "mixed":
        return str(rng.choice(["awgn", "rayleigh", "rician"]))
    if mode not in {"awgn", "rayleigh", "rician"}:
        raise ValueError("--channel_mode must be awgn, rayleigh, rician, or mixed.")
    return mode


def generate_sample(
    rng: np.random.Generator,
    cfg: NRPhyConfig,
    channel_mode: str,
    doppler_hz: float,
) -> dict[str, np.ndarray | str | float]:
    mapper = ResourceGridMapper(cfg, "DL")
    modem = OFDMModem(cfg)
    estimator = ChannelEstimator(cfg, mapper)
    data_re = sum(mapper.data_re_per_slot(slot_idx) for slot_idx in range(cfg.n_dl_slots))
    tx_bits = rng.integers(0, 2, size=2 * data_re, dtype=np.uint8)
    semantic_tx_symbols = qpsk_modulate(tx_bits)
    grids, _, allocations = mapper.map_symbols_to_slots(semantic_tx_symbols, cfg.n_dl_slots)
    tx_waveform = modem.modulate_frame(grids)

    channel = MultipathChannel(
        cfg,
        channel_type=channel_mode,
        delays=(0, 2, 5),
        powers_db=(0.0, -3.0, -8.0),
        rician_k_db=6.0,
        doppler_hz=doppler_hz,
        rng=rng,
    )
    channel_out = channel.transmit(tx_waveform)
    rx_grids = modem.demodulate_frame(channel_out.waveform, cfg.n_dl_slots)

    h_ls_slots = []
    equalized_slots = []
    pilot_masks = []
    pilot_obs = []
    for slot_idx, rx_grid in enumerate(rx_grids):
        h_ls = estimator.estimate_slot(rx_grid, slot_idx)
        h_ls_slots.append(h_ls)
        equalized_slots.append(
            estimator.equalize(rx_grid, h_ls, method="mmse", noise_variance=channel_out.noise_variance)
        )
        mask = mapper.pilot_mask_for_slot(slot_idx)
        pilot_masks.append(mask)
        pilot_obs.append(np.where(mask, rx_grid, 0.0))

    h_ls_grid = np.stack(h_ls_slots, axis=0)
    h_true = (
        channel_out.frequency_response_grid
        if channel_out.frequency_response_grid is not None
        else impulse_response_to_grid(cfg, channel_out.impulse_response, n_slots=cfg.n_dl_slots)
    )
    equalized_symbols = mapper.extract_data_symbols(equalized_slots)

    return {
        "rx_grid": np.stack(rx_grids, axis=0).astype(np.complex64),
        "pilot_obs": np.stack(pilot_obs, axis=0).astype(np.complex64),
        "pilot_mask": np.stack(pilot_masks, axis=0).astype(np.bool_),
        "H_ls_grid": h_ls_grid.astype(np.complex64),
        "H_true": h_true.astype(np.complex64),
        "equalized_symbols": equalized_symbols.astype(np.complex64),
        "semantic_tx_symbols": semantic_tx_symbols.astype(np.complex64),
        "allocation": np.stack(allocations, axis=0).astype(np.uint8),
        "channel_mode": channel_mode,
        "snr_db": np.float32(cfg.snr_db),
        "doppler": np.float32(doppler_hz),
        "feedback_bits": np.int32(0),
        "ls_nmse": np.float32(csi_nmse(h_true, h_ls_grid)),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--snr_min", type=float, default=0.0)
    parser.add_argument("--snr_max", type=float, default=30.0)
    parser.add_argument("--doppler_min", type=float, default=0.0)
    parser.add_argument("--doppler_max", type=float, default=0.0)
    parser.add_argument("--channel_mode", default="mixed")
    parser.add_argument("--output_path", default=str(PACKAGE_DIR / "outputs" / "comm_foundation_sanity_100.npz"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n_fft", type=int, default=128)
    parser.add_argument("--n_subcarriers", type=int, default=72)
    parser.add_argument("--dl_slots", type=int, default=1)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    rng = np.random.default_rng(args.seed)
    samples = []
    for _ in range(args.num_samples):
        snr_db = float(rng.uniform(args.snr_min, args.snr_max))
        doppler = float(rng.uniform(args.doppler_min, args.doppler_max))
        cfg = NRPhyConfig(
            n_fft=args.n_fft,
            n_subcarriers=args.n_subcarriers,
            n_dl_slots=args.dl_slots,
            n_ul_slots=1,
            snr_db=snr_db,
            rng_seed=int(rng.integers(0, 2**31 - 1)),
        )
        channel_mode = choose_channel_mode(rng, args.channel_mode)
        samples.append(generate_sample(rng, cfg, channel_mode, doppler))

    array_keys = [
        "rx_grid",
        "pilot_obs",
        "pilot_mask",
        "H_ls_grid",
        "H_true",
        "equalized_symbols",
        "semantic_tx_symbols",
        "allocation",
        "snr_db",
        "doppler",
        "feedback_bits",
        "ls_nmse",
    ]
    out = {key: np.stack([sample[key] for sample in samples], axis=0) for key in array_keys}
    out["channel_mode"] = np.asarray([sample["channel_mode"] for sample in samples])
    out["image_gt"] = np.asarray(["synthetic_qpsk_payload"] * args.num_samples)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **out)

    summary = {
        "num_samples": args.num_samples,
        "output_path": str(output_path),
        "shapes": {key: list(value.shape) for key, value in out.items()},
        "dtypes": {key: str(value.dtype) for key, value in out.items()},
        "mean_ls_nmse": float(np.mean(out["ls_nmse"])),
    }
    output_path.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
