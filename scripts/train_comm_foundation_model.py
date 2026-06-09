#!/usr/bin/env python3
"""Train the minimal complex communication foundation model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split

PACKAGE_DIR = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = PACKAGE_DIR.parent
sys.path.insert(0, str(PACKAGE_PARENT))

from nr_tdd_semantic.models.comm_foundation_model import (  # noqa: E402
    CommFoundationChannelEstimator,
    CommFoundationConfig,
    complex_np_to_channels,
    nmse_loss,
)


class CSIDataset(Dataset):
    def __init__(self, dataset_path: str):
        data = np.load(dataset_path)
        h_ls = data["H_ls_grid"].astype(np.complex64)
        h_true = data["H_true"].astype(np.complex64)
        self.x = h_ls.reshape(-1, *h_ls.shape[-2:])
        self.y = h_true.reshape(-1, *h_true.shape[-2:])
        self.ls_nmse = float(np.mean(np.abs(self.x - self.y) ** 2) / max(float(np.mean(np.abs(self.y) ** 2)), 1e-30))

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = complex_np_to_channels(self.x[idx : idx + 1]).squeeze(0)
        y = complex_np_to_channels(self.y[idx : idx + 1]).squeeze(0)
        return x, y

    def subset_ls_nmse(self, indices: list[int]) -> float:
        x = self.x[indices]
        y = self.y[indices]
        return float(np.mean(np.abs(x - y) ** 2) / max(float(np.mean(np.abs(y) ** 2)), 1e-30))


def masked_input(x: torch.Tensor, mask_prob: float = 0.25) -> tuple[torch.Tensor, torch.Tensor]:
    mask = (torch.rand_like(x[:, :1]) < mask_prob).float()
    mask = torch.cat([mask, mask], dim=1)
    return x * (1.0 - mask), mask


def reliability_target_from_error(pred: torch.Tensor, target: torch.Tensor, tau: float) -> torch.Tensor:
    err = torch.sum((pred - target) ** 2, dim=1, keepdim=True)
    return torch.exp(-err / max(float(tau), 1e-12)).detach()


def reliability_loss(model, x: torch.Tensor, pred: torch.Tensor, target: torch.Tensor, tau: float) -> torch.Tensor:
    z_comm = model.z_comm(x)
    reliability = model.reliability_head(z_comm)
    target_map = reliability_target_from_error(pred, target, tau)
    return torch.mean((reliability - target_map) ** 2)


def run_model_loss(model, x: torch.Tensor, y: torch.Tensor, args, phase: str) -> torch.Tensor:
    pred = model(x)
    loss = 0.0 * torch.mean(pred)
    if phase in {"ce", "joint"}:
        loss = loss + args.lambda_ce * nmse_loss(pred, y)

    if phase in {"pretrain", "joint"} and args.use_masked_csi:
        x_masked, mask = masked_input(x)
        pred_masked = model(x_masked)
        masked_mse = torch.sum(((pred_masked - y) ** 2) * mask) / mask.sum().clamp_min(1.0)
        target_power = torch.mean(y**2).clamp_min(1e-12)
        loss = loss + args.lambda_mask * masked_mse / target_power

    if phase in {"pretrain", "joint"} and args.use_denoising:
        noise_std = 0.08 * torch.sqrt(torch.mean(x**2)).detach()
        x_noisy = x + noise_std * torch.randn_like(x)
        loss = loss + args.lambda_denoise * nmse_loss(model(x_noisy), y)

    if args.use_reliability:
        loss = loss + args.lambda_reliability * reliability_loss(model, x, pred, y, args.reliability_tau)
    return loss


def train_one_epoch(model, loader, optimizer, args, device, phase: str) -> float:
    model.train()
    losses = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        loss = run_model_loss(model, x, y, args, phase)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


@torch.no_grad()
def evaluate(model, loader, device, use_reliability: bool = False, reliability_tau: float = 0.01) -> dict[str, float]:
    model.eval()
    losses = []
    reliability_values = []
    error_values = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        losses.append(float(nmse_loss(pred, y).cpu()))
        if use_reliability:
            reliability = model.reliability_head(model.z_comm(x)).detach().cpu().numpy().reshape(-1)
            err = torch.sum((pred - y) ** 2, dim=1).detach().cpu().numpy().reshape(-1)
            reliability_values.append(reliability)
            error_values.append(err)
    result = {"val_nmse": float(np.mean(losses))}
    if use_reliability and reliability_values:
        rel = np.concatenate(reliability_values)
        err = np.concatenate(error_values)
        if np.std(rel) > 1e-12 and np.std(err) > 1e-12:
            result["reliability_error_corr"] = float(np.corrcoef(rel, err)[0, 1])
        else:
            result["reliability_error_corr"] = float("nan")
        high = rel >= np.quantile(rel, 0.75)
        low = rel <= np.quantile(rel, 0.25)
        result["high_reliability_error"] = float(np.mean(err[high]))
        result["low_reliability_error"] = float(np.mean(err[low]))
        target = np.exp(-err / max(float(reliability_tau), 1e-12))
        result["reliability_target_mse"] = float(np.mean((rel - target) ** 2))
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--eval_dataset_paths", default="", help="Comma-separated held-out npz files for post-training evaluation.")
    parser.add_argument("--output_dir", default=str(PACKAGE_DIR / "outputs" / "comm_foundation_ckpt"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--use_masked_csi", action="store_true")
    parser.add_argument("--use_denoising", action="store_true")
    parser.add_argument(
        "--training_strategy",
        choices=("scratch_ce", "pretrain_then_finetune", "joint_pretrain_ce"),
        default="joint_pretrain_ce",
    )
    parser.add_argument("--sample_fraction", type=float, default=1.0, help="Use 0.01, 0.05, 0.10, or 1.0 for few-shot studies.")
    parser.add_argument("--pretrain_epochs", type=int, default=3)
    parser.add_argument("--lambda_mask", type=float, default=1.0)
    parser.add_argument("--lambda_denoise", type=float, default=1.0)
    parser.add_argument("--lambda_ce", type=float, default=1.0)
    parser.add_argument("--use_reliability", action="store_true")
    parser.add_argument("--lambda_reliability", type=float, default=0.2)
    parser.add_argument("--reliability_tau", type=float, default=0.01)
    parser.add_argument("--hidden_channels", type=int, default=16)
    parser.add_argument("--depth", type=int, default=4)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = CSIDataset(args.dataset_path)
    if not (0.0 < args.sample_fraction <= 1.0):
        raise ValueError("--sample_fraction must be in (0, 1].")
    base_indices = np.arange(len(dataset))
    rng = np.random.default_rng(2026)
    rng.shuffle(base_indices)
    used_count = max(2, int(round(len(dataset) * args.sample_fraction)))
    subset = Subset(dataset, base_indices[:used_count].tolist())
    val_size = max(1, int(0.2 * len(subset)))
    train_size = len(subset) - val_size
    train_ds, val_ds = random_split(subset, [train_size, val_size], generator=torch.Generator().manual_seed(2026))
    val_indices = [subset.indices[i] for i in val_ds.indices]
    val_ls_nmse = dataset.subset_ls_nmse(val_indices)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = CommFoundationConfig(hidden_complex_channels=args.hidden_channels, depth=args.depth)
    model = CommFoundationChannelEstimator(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = []
    best_path = output_dir / "best_comm_foundation_channel_estimator.pt"
    initial_eval = evaluate(model, val_loader, device, args.use_reliability, args.reliability_tau)
    best_val = initial_eval["val_nmse"]
    torch.save(model.checkpoint_payload(), best_path)
    history.append(
        {
            "epoch": 0,
            "phase": "identity",
            "train_loss": None,
            **initial_eval,
            "ls_baseline_nmse": dataset.ls_nmse,
            "val_ls_baseline_nmse": val_ls_nmse,
            "note": "identity_safe_initialization",
        }
    )

    if args.training_strategy == "scratch_ce":
        schedule = [("ce", args.epochs)]
    elif args.training_strategy == "pretrain_then_finetune":
        args.use_masked_csi = True
        args.use_denoising = True
        schedule = [("pretrain", args.pretrain_epochs), ("ce", args.epochs)]
    else:
        args.use_masked_csi = True
        args.use_denoising = True
        schedule = [("joint", args.epochs)]

    global_epoch = 0
    for phase, phase_epochs in schedule:
        for _ in range(phase_epochs):
            global_epoch += 1
            train_loss = train_one_epoch(model, train_loader, optimizer, args, device, phase)
            eval_result = evaluate(model, val_loader, device, args.use_reliability, args.reliability_tau)
            val_nmse = eval_result["val_nmse"]
            row = {
                "epoch": global_epoch,
                "phase": phase,
                "train_loss": train_loss,
                **eval_result,
                "ls_baseline_nmse": dataset.ls_nmse,
                "val_ls_baseline_nmse": val_ls_nmse,
                "improvement_over_val_ls_nmse": val_ls_nmse - val_nmse,
            }
            history.append(row)
            if val_nmse < best_val:
                best_val = val_nmse
                torch.save(model.checkpoint_payload(), best_path)
            print(json.dumps(row))

    summary = {
        "dataset_path": args.dataset_path,
        "checkpoint": str(best_path),
        "training_strategy": args.training_strategy,
        "sample_fraction": args.sample_fraction,
        "used_samples": used_count,
        "best_val_nmse": best_val,
        "best_val_nmse_db": 10.0 * np.log10(max(best_val, 1e-30)),
        "ls_baseline_nmse": dataset.ls_nmse,
        "ls_baseline_nmse_db": 10.0 * np.log10(max(dataset.ls_nmse, 1e-30)),
        "val_ls_baseline_nmse": val_ls_nmse,
        "val_ls_baseline_nmse_db": 10.0 * np.log10(max(val_ls_nmse, 1e-30)),
        "history": history,
    }
    best_payload = torch.load(best_path, map_location=device)
    model.load_state_dict(best_payload["state_dict"], strict=False)
    eval_paths = [item.strip() for item in args.eval_dataset_paths.split(",") if item.strip()]
    heldout = {}
    for eval_path in eval_paths:
        eval_dataset = CSIDataset(eval_path)
        eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size)
        heldout[Path(eval_path).stem] = evaluate(model, eval_loader, device, args.use_reliability, args.reliability_tau)
    if heldout:
        summary["heldout"] = heldout
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
