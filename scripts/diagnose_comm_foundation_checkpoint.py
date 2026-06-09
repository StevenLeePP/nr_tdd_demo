#!/usr/bin/env python3
"""Diagnose whether a comm-foundation checkpoint learned a nonzero residual."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PACKAGE_DIR = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = PACKAGE_DIR.parent
sys.path.insert(0, str(PACKAGE_PARENT))

from nr_tdd_semantic.config import NRPhyConfig  # noqa: E402
from nr_tdd_semantic.dsp import delay_domain_denoise_csi  # noqa: E402
from nr_tdd_semantic.models.comm_foundation_model import (  # noqa: E402
    CommFoundationChannelEstimator,
    CommFoundationConfig,
    complex_np_to_channels,
)


def nmse_np(reference: np.ndarray, estimate: np.ndarray) -> float:
    return float(np.mean(np.abs(reference - estimate) ** 2) / max(float(np.mean(np.abs(reference) ** 2)), 1e-30))


def tensor_stats(x: torch.Tensor) -> dict[str, float]:
    x = x.detach().float().cpu()
    return {
        "mean": float(torch.mean(x)),
        "var": float(torch.var(x, unbiased=False)),
        "max_abs": float(torch.max(torch.abs(x))),
        "l2_norm": float(torch.linalg.vector_norm(x)),
    }


def load_model(checkpoint_path: str | None, device: torch.device) -> CommFoundationChannelEstimator:
    if checkpoint_path:
        payload = torch.load(checkpoint_path, map_location=device)
        cfg = CommFoundationConfig(**payload["model_config"])
        model = CommFoundationChannelEstimator(cfg).to(device)
        model.load_state_dict(payload["state_dict"], strict=False)
    else:
        model = CommFoundationChannelEstimator(CommFoundationConfig()).to(device)
    model.eval()
    return model


def state_dict_delta(model_a: CommFoundationChannelEstimator, model_b: CommFoundationChannelEstimator) -> dict[str, float]:
    rows = {}
    total_sq = 0.0
    total_params = 0
    for key, tensor_a in model_a.state_dict().items():
        tensor_b = model_b.state_dict().get(key)
        if tensor_b is None or tensor_a.shape != tensor_b.shape:
            continue
        delta = (tensor_a.detach().cpu().float() - tensor_b.detach().cpu().float()).reshape(-1)
        total_sq += float(torch.sum(delta**2))
        total_params += int(delta.numel())
        rows[key] = float(torch.linalg.vector_norm(delta))
    return {
        "total_l2": float(total_sq**0.5),
        "mean_l2_per_param": float((total_sq / max(total_params, 1)) ** 0.5),
        "matched_tensors": len(rows),
        "per_tensor_l2": rows,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--epoch0_checkpoint", default=None)
    parser.add_argument("--output_dir", default=str(PACKAGE_DIR / "outputs" / "checkpoint_diagnostics"))
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--n_fft", type=int, default=128)
    parser.add_argument("--time_average", action="store_true", default=True)
    parser.add_argument("--no_time_average", dest="time_average", action="store_false")
    return parser


@torch.no_grad()
def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(args.dataset_path)
    h_ls = data["H_ls_grid"].astype(np.complex64).reshape(-1, *data["H_ls_grid"].shape[-2:])
    h_true = data["H_true"].astype(np.complex64).reshape(-1, *data["H_true"].shape[-2:])
    take = min(args.batch_size, h_ls.shape[0])
    h_ls = h_ls[:take]
    h_true = h_true[:take]
    cfg = NRPhyConfig(n_fft=args.n_fft, n_subcarriers=h_ls.shape[-2], n_dl_slots=1, n_ul_slots=1)
    h_structured = np.stack(
        [delay_domain_denoise_csi(sample, cfg, time_average=args.time_average) for sample in h_ls],
        axis=0,
    ).astype(np.complex64)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x = complex_np_to_channels(h_structured).to(device)
    y = complex_np_to_channels(h_true).to(device)
    untrained = load_model(None, device)
    trained = load_model(args.checkpoint, device)
    epoch0 = load_model(args.epoch0_checkpoint, device) if args.epoch0_checkpoint else None

    trained_residual = trained.residual(x)
    untrained_residual = untrained.residual(x)
    trained_hat = trained(x)
    untrained_hat = untrained(x)
    z_comm = trained.z_comm(x)
    h_structured_channels = complex_np_to_channels(h_structured).to(device)

    result = {
        "checkpoint": args.checkpoint,
        "dataset_path": args.dataset_path,
        "batch_size": take,
        "residual_stats_trained": tensor_stats(trained_residual),
        "residual_stats_untrained": tensor_stats(untrained_residual),
        "z_comm_stats_trained": tensor_stats(z_comm),
        "h_hat_vs_h_structured_nmse": float(torch.mean((trained_hat - h_structured_channels) ** 2) / torch.mean(h_structured_channels**2).clamp_min(1e-12)),
        "ls_smoothing_nmse": float(torch.mean((h_structured_channels - y) ** 2) / torch.mean(y**2).clamp_min(1e-12)),
        "comm_foundation_untrained_nmse": float(torch.mean((untrained_hat - y) ** 2) / torch.mean(y**2).clamp_min(1e-12)),
        "comm_foundation_trained_nmse": float(torch.mean((trained_hat - y) ** 2) / torch.mean(y**2).clamp_min(1e-12)),
        "trained_vs_untrained_param_delta": state_dict_delta(trained, untrained),
    }
    if epoch0 is not None:
        result["trained_vs_epoch0_param_delta"] = state_dict_delta(trained, epoch0)

    json_path = output_dir / "checkpoint_diagnostics.json"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    md_lines = [
        "# Checkpoint Diagnostics",
        "",
        f"- Checkpoint: `{args.checkpoint}`",
        f"- Dataset: `{args.dataset_path}`",
        f"- Batch size: `{take}`",
        "",
        "## NMSE",
        "",
        f"- LS smoothing NMSE: `{result['ls_smoothing_nmse']:.6g}`",
        f"- Untrained NMSE: `{result['comm_foundation_untrained_nmse']:.6g}`",
        f"- Trained NMSE: `{result['comm_foundation_trained_nmse']:.6g}`",
        f"- H_hat vs H_structured NMSE: `{result['h_hat_vs_h_structured_nmse']:.6g}`",
        "",
        "## Residual Stats",
        "",
        f"- Trained residual L2: `{result['residual_stats_trained']['l2_norm']:.6g}`",
        f"- Trained residual max abs: `{result['residual_stats_trained']['max_abs']:.6g}`",
        f"- Parameter delta vs untrained L2: `{result['trained_vs_untrained_param_delta']['total_l2']:.6g}`",
    ]
    (output_dir / "checkpoint_diagnostics.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
