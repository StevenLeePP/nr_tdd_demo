#!/usr/bin/env python3
"""Run training strategy/few-shot comparisons and collect CSV/Markdown results."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

PACKAGE_DIR = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = PACKAGE_DIR.parent
sys.path.insert(0, str(PACKAGE_PARENT))
TRAIN_SCRIPT = PACKAGE_DIR / "scripts" / "train_comm_foundation_model.py"

from nr_tdd_semantic.config import NRPhyConfig  # noqa: E402
from nr_tdd_semantic.dsp import delay_domain_denoise_csi  # noqa: E402


def parse_list(text: str, cast):
    return [cast(item.strip()) for item in text.split(",") if item.strip()]


def nmse_db(value: float) -> float:
    return float(10.0 * np.log10(max(float(value), 1e-30)))


def load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def ls_smoothing_nmse_db(dataset_path: Path, n_fft: int) -> float:
    data = np.load(dataset_path)
    h_ls = data["H_ls_grid"].astype(np.complex64).reshape(-1, *data["H_ls_grid"].shape[-2:])
    h_true = data["H_true"].astype(np.complex64).reshape(-1, *data["H_true"].shape[-2:])
    cfg = NRPhyConfig(n_fft=n_fft, n_subcarriers=h_ls.shape[-2], n_dl_slots=1, n_ul_slots=1)
    structured = np.stack([delay_domain_denoise_csi(sample, cfg, time_average=True) for sample in h_ls], axis=0)
    nmse = float(np.mean(np.abs(h_true - structured) ** 2) / max(float(np.mean(np.abs(h_true) ** 2)), 1e-30))
    return nmse_db(nmse)


def best_epoch(summary: dict) -> int:
    best = summary.get("best_val_nmse")
    for row in summary.get("history", []):
        if abs(float(row.get("val_nmse", float("inf"))) - float(best)) < 1e-12:
            return int(row.get("epoch", -1))
    return -1


def run_case(args, strategy: str, fraction: float, eval_paths: list[str]) -> dict:
    tag = f"{strategy}_{fraction:g}".replace(".", "p")
    out_dir = Path(args.output_dir) / "checkpoints" / tag
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--dataset_path",
        str(Path(args.dataset_dir) / "train.npz"),
        "--eval_dataset_paths",
        ",".join(eval_paths),
        "--output_dir",
        str(out_dir),
        "--training_strategy",
        strategy,
        "--sample_fraction",
        str(fraction),
        "--epochs",
        str(args.epochs),
        "--pretrain_epochs",
        str(args.pretrain_epochs),
        "--batch_size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
    ]
    if args.use_reliability:
        cmd.append("--use_reliability")
    if args.dry_run:
        return {"training_strategy": strategy, "sample_fraction": fraction, "command": " ".join(cmd)}
    subprocess.run(cmd, cwd=PACKAGE_DIR, check=True)
    summary = load_summary(out_dir / "training_summary.json")
    heldout = summary.get("heldout", {})
    return {
        "training_strategy": strategy,
        "sample_fraction": fraction,
        "val_nmse_db": nmse_db(summary["best_val_nmse"]),
        "unseen_snr_nmse_db": nmse_db(heldout.get("test_unseen_snr", {}).get("val_nmse", float("nan"))),
        "unseen_doppler_nmse_db": nmse_db(heldout.get("test_unseen_doppler", {}).get("val_nmse", float("nan"))),
        "unseen_delay_nmse_db": nmse_db(heldout.get("test_unseen_delay", {}).get("val_nmse", float("nan"))),
        "unseen_rician_k_nmse_db": nmse_db(heldout.get("test_unseen_rician_k", {}).get("val_nmse", float("nan"))),
        "ls_baseline_nmse_db": summary["ls_baseline_nmse_db"],
        "ls_smoothing_baseline_nmse_db": ls_smoothing_nmse_db(Path(args.dataset_dir) / "val.npz", args.n_fft),
        "best_epoch": best_epoch(summary),
        "checkpoint_path": summary["checkpoint"],
        "training_summary_path": str(out_dir / "training_summary.json"),
    }


def write_outputs(rows: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "training_results.csv"
    fields = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    valid = [row for row in rows if "command" not in row]
    lines = ["# Training Comparison Summary", "", f"- Results CSV: `{csv_path}`", ""]
    if valid:
        lines.append("| Strategy | Fraction | Val NMSE | Unseen SNR | Unseen Doppler | Unseen Delay | Unseen Rician K | Best Epoch |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for row in valid:
            lines.append(
                f"| {row['training_strategy']} | {row['sample_fraction']} | {row['val_nmse_db']:.2f} | "
                f"{row['unseen_snr_nmse_db']:.2f} | {row['unseen_doppler_nmse_db']:.2f} | "
                f"{row['unseen_delay_nmse_db']:.2f} | {row['unseen_rician_k_nmse_db']:.2f} | {row['best_epoch']} |"
            )
        lines.extend(
            [
                "",
                "## Key Questions",
                "",
                "- 预训练是否优于 scratch_ce：根据同一 sample_fraction 下的 val/held-out NMSE 对比判断。",
                "- 少样本优势：重点看 0.01、0.05、0.10 三档。",
                "- held-out 泛化：重点看 unseen SNR/Doppler/delay/Rician K 四列。",
                "- Z_comm 可迁移性：只有当预训练策略在少样本和 held-out 上稳定优于 scratch 时才成立。",
            ]
        )
    else:
        lines.append("Dry-run only; no metrics collected.")
    (output_dir / "training_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", default=str(PACKAGE_DIR / "outputs" / "dataset_v1"))
    parser.add_argument("--output_dir", default=str(PACKAGE_DIR / "outputs" / "training_comparison"))
    parser.add_argument("--strategies", default="scratch_ce,pretrain_then_finetune,joint_pretrain_ce")
    parser.add_argument("--sample_fractions", default="0.01,0.05,0.10,1.0")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--pretrain_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n_fft", type=int, default=128)
    parser.add_argument("--use_reliability", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    dataset_dir = Path(args.dataset_dir)
    eval_paths = [
        str(dataset_dir / "test_unseen_snr.npz"),
        str(dataset_dir / "test_unseen_doppler.npz"),
        str(dataset_dir / "test_unseen_delay.npz"),
        str(dataset_dir / "test_unseen_rician_k.npz"),
    ]
    rows = []
    for strategy in parse_list(args.strategies, str):
        for fraction in parse_list(args.sample_fractions, float):
            rows.append(run_case(args, strategy, fraction, eval_paths))
    write_outputs(rows, Path(args.output_dir))
    print(json.dumps({"cases": len(rows), "output_dir": args.output_dir}, indent=2))


if __name__ == "__main__":
    main()
