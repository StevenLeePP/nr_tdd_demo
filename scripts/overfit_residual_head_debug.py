#!/usr/bin/env python3
"""Overfit a tiny batch to verify residual-branch trainability."""

from __future__ import annotations

import argparse
import copy
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
    nmse_loss,
)


def module_param_norm(module: torch.nn.Module) -> float:
    total = 0.0
    for param in module.parameters():
        total += float(torch.sum(param.detach().float() ** 2))
    return float(total**0.5)


def module_grad_norm(module: torch.nn.Module) -> float:
    total = 0.0
    for param in module.parameters():
        if param.grad is not None:
            total += float(torch.sum(param.grad.detach().float() ** 2))
    return float(total**0.5)


def tensor_l2(x: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(x.detach().float()).cpu())


def load_batch(dataset_path: str, samples: int, n_fft: int, time_average: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    data = np.load(dataset_path)
    h_ls = data["H_ls_grid"].astype(np.complex64).reshape(-1, *data["H_ls_grid"].shape[-2:])[:samples]
    h_true = data["H_true"].astype(np.complex64).reshape(-1, *data["H_true"].shape[-2:])[:samples]
    cfg = NRPhyConfig(n_fft=n_fft, n_subcarriers=h_ls.shape[-2], n_dl_slots=1, n_ul_slots=1)
    h_structured = np.stack([delay_domain_denoise_csi(x, cfg, time_average=time_average) for x in h_ls], axis=0)
    return complex_np_to_channels(h_ls), complex_np_to_channels(h_structured.astype(np.complex64)), complex_np_to_channels(h_true)


def set_trainable(model: CommFoundationChannelEstimator, train_backbone: bool) -> None:
    for param in model.parameters():
        param.requires_grad_(False)
    for param in model.channel_head.parameters():
        param.requires_grad_(True)
    model.residual_scale_param.requires_grad_(True)
    if train_backbone:
        for param in model.backbone.parameters():
            param.requires_grad_(True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_path", default=str(PACKAGE_DIR / "outputs" / "comm_foundation_sanity_100.npz"))
    parser.add_argument("--output_dir", default=str(PACKAGE_DIR / "outputs" / "residual_overfit_debug"))
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--n_fft", type=int, default=128)
    parser.add_argument("--hidden_channels", type=int, default=16)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--train_backbone", action="store_true")
    parser.add_argument("--time_average", action="store_true", default=True)
    parser.add_argument("--no_time_average", dest="time_average", action="store_false")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _, x_structured, y = load_batch(args.dataset_path, args.samples, args.n_fft, args.time_average)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x_structured = x_structured.to(device)
    y = y.to(device)

    model = CommFoundationChannelEstimator(
        CommFoundationConfig(hidden_complex_channels=args.hidden_channels, depth=args.depth)
    ).to(device)
    set_trainable(model, train_backbone=args.train_backbone)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    with torch.no_grad():
        h_structured_nmse = float(nmse_loss(x_structured, y).cpu())
        before_hat = model(x_structured)
        before_residual = model.residual(x_structured)
        before_h_hat_nmse = float(nmse_loss(before_hat, y).cpu())
        before_residual_norm = tensor_l2(before_residual)
        before_head_norm = module_param_norm(model.channel_head)
        before_backbone_norm = module_param_norm(model.backbone)
        before_scale = float(model.residual_scale_param.detach().cpu())

    curve = []
    last_grad = {}
    best_loss = float("inf")
    best_step = 0
    best_state = copy.deepcopy(model.state_dict())
    for step in range(1, args.steps + 1):
        optimizer.zero_grad()
        pred = model(x_structured)
        loss = nmse_loss(pred, y)
        loss.backward()
        last_grad = {
            "backbone_grad_norm": module_grad_norm(model.backbone),
            "channel_head_grad_norm": module_grad_norm(model.channel_head),
            "residual_scale_grad_norm": 0.0
            if model.residual_scale_param.grad is None
            else float(torch.abs(model.residual_scale_param.grad.detach()).cpu()),
        }
        optimizer.step()
        loss_value = float(loss.detach().cpu())
        if loss_value < best_loss:
            best_loss = loss_value
            best_step = step
            best_state = copy.deepcopy(model.state_dict())
        if step == 1 or step % max(1, args.steps // 20) == 0 or step == args.steps:
            curve.append({"step": step, "train_nmse": loss_value, **last_grad})

    model.load_state_dict(best_state)
    with torch.no_grad():
        after_hat = model(x_structured)
        after_residual = model.residual(x_structured)
        after_h_hat_nmse = float(nmse_loss(after_hat, y).cpu())
        after_residual_norm = tensor_l2(after_residual)
        after_head_norm = module_param_norm(model.channel_head)
        after_backbone_norm = module_param_norm(model.backbone)
        after_scale = float(model.residual_scale_param.detach().cpu())

    checkpoint_path = output_dir / "overfit_residual_debug_checkpoint.pt"
    torch.save(model.checkpoint_payload(), checkpoint_path)
    reloaded = CommFoundationChannelEstimator(model.cfg).to(device)
    payload = torch.load(checkpoint_path, map_location=device)
    reloaded.load_state_dict(payload["state_dict"], strict=False)
    reloaded.eval()
    with torch.no_grad():
        reload_max_abs_diff = float(torch.max(torch.abs(reloaded(x_structured) - after_hat)).cpu())

    result = {
        "dataset_path": args.dataset_path,
        "samples": int(x_structured.shape[0]),
        "steps": args.steps,
        "lr": args.lr,
        "train_backbone": args.train_backbone,
        "best_step": best_step,
        "best_train_nmse": best_loss,
        "h_structured_nmse": h_structured_nmse,
        "h_hat_nmse_before": before_h_hat_nmse,
        "h_hat_nmse_after": after_h_hat_nmse,
        "h_hat_gain_over_structured_nmse": h_structured_nmse - after_h_hat_nmse,
        "residual_norm_before": before_residual_norm,
        "residual_norm_after": after_residual_norm,
        "channel_head_param_norm_before": before_head_norm,
        "channel_head_param_norm_after": after_head_norm,
        "backbone_param_norm_before": before_backbone_norm,
        "backbone_param_norm_after": after_backbone_norm,
        "residual_scale_before": before_scale,
        "residual_scale_after": after_scale,
        "last_gradient_norms": last_grad,
        "reload_max_abs_diff": reload_max_abs_diff,
        "checkpoint_path": str(checkpoint_path),
        "train_curve": curve,
        "can_overfit": bool(after_h_hat_nmse < 0.5 * h_structured_nmse and after_residual_norm > 1e-6),
    }
    (output_dir / "overfit_residual_debug.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = [
        "# Residual Overfit Debug",
        "",
        f"- Samples: `{result['samples']}`",
        f"- Best step / train NMSE: `{best_step}` / `{best_loss:.6g}`",
        f"- Structured NMSE: `{h_structured_nmse:.6g}`",
        f"- H_hat NMSE before: `{before_h_hat_nmse:.6g}`",
        f"- H_hat NMSE after: `{after_h_hat_nmse:.6g}`",
        f"- Gain over structured NMSE: `{result['h_hat_gain_over_structured_nmse']:.6g}`",
        f"- Residual norm before/after: `{before_residual_norm:.6g}` / `{after_residual_norm:.6g}`",
        f"- Channel head norm before/after: `{before_head_norm:.6g}` / `{after_head_norm:.6g}`",
        f"- Backbone norm before/after: `{before_backbone_norm:.6g}` / `{after_backbone_norm:.6g}`",
        f"- Residual scale before/after: `{before_scale:.6g}` / `{after_scale:.6g}`",
        f"- Last gradient norms: `{last_grad}`",
        f"- Reload max abs diff: `{reload_max_abs_diff:.6g}`",
        f"- Can overfit: `{result['can_overfit']}`",
    ]
    (output_dir / "overfit_residual_debug.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
