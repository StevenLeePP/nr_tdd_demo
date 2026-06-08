#!/usr/bin/env python3
"""Train the minimal complex communication foundation model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

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


def masked_input(x: torch.Tensor, mask_prob: float = 0.25) -> tuple[torch.Tensor, torch.Tensor]:
    mask = (torch.rand_like(x[:, :1]) < mask_prob).float()
    mask = torch.cat([mask, mask], dim=1)
    return x * (1.0 - mask), mask


def train_one_epoch(model, loader, optimizer, args, device) -> float:
    model.train()
    losses = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        loss = args.lambda_ce * nmse_loss(model(x), y)

        if args.use_masked_csi:
            x_masked, mask = masked_input(x)
            pred_masked = model(x_masked)
            masked_mse = torch.sum(((pred_masked - y) ** 2) * mask) / mask.sum().clamp_min(1.0)
            target_power = torch.mean(y**2).clamp_min(1e-12)
            loss = loss + args.lambda_mask * masked_mse / target_power

        if args.use_denoising:
            noise_std = 0.08 * torch.sqrt(torch.mean(x**2)).detach()
            x_noisy = x + noise_std * torch.randn_like(x)
            loss = loss + args.lambda_denoise * nmse_loss(model(x_noisy), y)

        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    losses = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        losses.append(float(nmse_loss(model(x), y).cpu()))
    return float(np.mean(losses))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--output_dir", default=str(PACKAGE_DIR / "outputs" / "comm_foundation_ckpt"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--use_masked_csi", action="store_true")
    parser.add_argument("--use_denoising", action="store_true")
    parser.add_argument("--lambda_mask", type=float, default=1.0)
    parser.add_argument("--lambda_denoise", type=float, default=1.0)
    parser.add_argument("--lambda_ce", type=float, default=1.0)
    parser.add_argument("--hidden_channels", type=int, default=16)
    parser.add_argument("--depth", type=int, default=4)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = CSIDataset(args.dataset_path)
    val_size = max(1, int(0.2 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(2026))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = CommFoundationConfig(hidden_complex_channels=args.hidden_channels, depth=args.depth)
    model = CommFoundationChannelEstimator(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = []
    best_val = float("inf")
    best_path = output_dir / "best_comm_foundation_channel_estimator.pt"
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, args, device)
        val_nmse = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": train_loss, "val_nmse": val_nmse, "ls_baseline_nmse": dataset.ls_nmse}
        history.append(row)
        if val_nmse < best_val:
            best_val = val_nmse
            torch.save(model.checkpoint_payload(), best_path)
        print(json.dumps(row))

    summary = {
        "dataset_path": args.dataset_path,
        "checkpoint": str(best_path),
        "best_val_nmse": best_val,
        "best_val_nmse_db": 10.0 * np.log10(max(best_val, 1e-30)),
        "ls_baseline_nmse": dataset.ls_nmse,
        "ls_baseline_nmse_db": 10.0 * np.log10(max(dataset.ls_nmse, 1e-30)),
        "history": history,
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
