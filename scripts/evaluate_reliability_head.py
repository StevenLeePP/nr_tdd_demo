#!/usr/bin/env python3
"""Evaluate ReliabilityHead on validation and held-out CSI datasets."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

PACKAGE_DIR = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = PACKAGE_DIR.parent
sys.path.insert(0, str(PACKAGE_PARENT))

from nr_tdd_semantic.models.comm_foundation_model import (  # noqa: E402
    CommFoundationChannelEstimator,
    CommFoundationConfig,
    complex_np_to_channels,
)


def load_model(checkpoint_path: str, device: torch.device) -> CommFoundationChannelEstimator:
    payload = torch.load(checkpoint_path, map_location=device)
    cfg = CommFoundationConfig(**payload["model_config"])
    model = CommFoundationChannelEstimator(cfg).to(device)
    model.load_state_dict(payload["state_dict"], strict=False)
    model.eval()
    return model


def load_dataset(path: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    h_ls = data["H_ls_grid"].astype(np.complex64).reshape(-1, *data["H_ls_grid"].shape[-2:])
    h_true = data["H_true"].astype(np.complex64).reshape(-1, *data["H_true"].shape[-2:])
    return h_ls, h_true


@torch.no_grad()
def evaluate_dataset(model, dataset_path: str, device: torch.device, batch_size: int, tau: float) -> dict:
    h_ls, h_true = load_dataset(dataset_path)
    reliability_values = []
    error_values = []
    for start in range(0, h_ls.shape[0], batch_size):
        x = complex_np_to_channels(h_ls[start : start + batch_size]).to(device)
        y = complex_np_to_channels(h_true[start : start + batch_size]).to(device)
        pred = model(x)
        reliability = model.reliability_head(model.z_comm(x)).detach().cpu().numpy().reshape(-1)
        err = torch.sum((pred - y) ** 2, dim=1).detach().cpu().numpy().reshape(-1)
        reliability_values.append(reliability)
        error_values.append(err)
    rel = np.concatenate(reliability_values)
    err = np.concatenate(error_values)
    high = rel >= np.quantile(rel, 0.75)
    low = rel <= np.quantile(rel, 0.25)
    target = np.exp(-err / max(float(tau), 1e-12))
    corr = float(np.corrcoef(rel, err)[0, 1]) if np.std(rel) > 1e-12 and np.std(err) > 1e-12 else float("nan")
    high_err = float(np.mean(err[high]))
    low_err = float(np.mean(err[low]))
    return {
        "dataset": Path(dataset_path).stem,
        "dataset_path": dataset_path,
        "num_samples": int(h_ls.shape[0]),
        "reliability_error_corr": corr,
        "high_reliability_error": high_err,
        "low_reliability_error": low_err,
        "high_low_error_ratio": float(high_err / max(low_err, 1e-30)),
        "reliability_target_mse": float(np.mean((rel - target) ** 2)),
    }


def maybe_save_heatmaps(model, dataset_path: str, output_dir: Path, device: torch.device, count: int) -> None:
    if count <= 0:
        return
    import matplotlib.pyplot as plt

    h_ls, _ = load_dataset(dataset_path)
    take = min(count, h_ls.shape[0])
    x = complex_np_to_channels(h_ls[:take]).to(device)
    with torch.no_grad():
        maps = model.reliability_head(model.z_comm(x)).detach().cpu().numpy()[:, 0]
    heatmap_dir = output_dir / "heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    for idx, rel_map in enumerate(maps):
        plt.figure(figsize=(6, 3))
        plt.imshow(rel_map, aspect="auto", origin="lower", cmap="viridis", vmin=0.0, vmax=1.0)
        plt.colorbar(label="Reliability")
        plt.xlabel("OFDM symbol")
        plt.ylabel("Subcarrier")
        plt.tight_layout()
        plt.savefig(heatmap_dir / f"{Path(dataset_path).stem}_sample_{idx}.png", dpi=160)
        plt.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset_paths", required=True, help="Comma-separated val/held-out npz files.")
    parser.add_argument("--output_dir", default=str(PACKAGE_DIR / "outputs" / "reliability_eval"))
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--save_heatmaps", type=int, default=0)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    dataset_paths = [item.strip() for item in args.dataset_paths.split(",") if item.strip()]
    rows = [evaluate_dataset(model, path, device, args.batch_size, args.tau) for path in dataset_paths]
    csv_path = output_dir / "reliability_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    if args.save_heatmaps and dataset_paths:
        maybe_save_heatmaps(model, dataset_paths[0], output_dir, device, args.save_heatmaps)
    (output_dir / "reliability_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps({"rows": rows, "output": str(csv_path)}, indent=2))


if __name__ == "__main__":
    main()
