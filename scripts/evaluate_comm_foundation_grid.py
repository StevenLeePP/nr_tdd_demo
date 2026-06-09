#!/usr/bin/env python3
"""Run estimator ablations over SNR/Doppler/channel grids."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parents[1]
RUN_DEMO = PACKAGE_DIR / "run_demo.py"


def parse_list(text: str, cast):
    return [cast(item.strip()) for item in text.split(",") if item.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimators", default="ls,ls_smoothing,comm_foundation_untrained,comm_foundation_trained")
    parser.add_argument("--snrs", default="10,20")
    parser.add_argument("--dopplers", default="0")
    parser.add_argument("--channels", default="rayleigh,rician")
    parser.add_argument("--pilot-spacings", default="4")
    parser.add_argument("--checkpoint", default="outputs/comm_foundation_ckpt_residual_safe/best_comm_foundation_channel_estimator.pt")
    parser.add_argument("--output-dir", default=str(PACKAGE_DIR / "outputs" / "grid_eval"))
    parser.add_argument("--semantic", choices=("swinjscc", "fallback"), default="fallback")
    parser.add_argument("--dl-slots", type=int, default=12)
    parser.add_argument("--ul-slots", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def run_case(args, estimator: str, channel: str, snr: float, doppler: float, pilot_spacing: int) -> dict:
    snr_tag = f"{snr:g}dB".replace(".", "p")
    doppler_tag = f"{doppler:g}Hz".replace(".", "p")
    case_dir = Path(args.output_dir) / f"{channel}_snr_{snr_tag}_doppler_{doppler_tag}_pilot_{pilot_spacing}_{estimator}"
    cmd = [
        sys.executable,
        str(RUN_DEMO),
        "--channel",
        channel,
        "--snr-db",
        str(snr),
        "--doppler-hz",
        str(doppler),
        "--pilot-spacing",
        str(pilot_spacing),
        "--channel-estimator",
        estimator,
        "--semantic",
        args.semantic,
        "--dl-slots",
        str(args.dl_slots),
        "--ul-slots",
        str(args.ul_slots),
        "--seed",
        str(args.seed),
        "--output-dir",
        str(case_dir),
    ]
    if estimator == "comm_foundation_trained":
        cmd.extend(["--comm-foundation-checkpoint", args.checkpoint])
    if args.dry_run:
        return {"command": " ".join(cmd)}
    subprocess.run(cmd, cwd=PACKAGE_DIR, check=True)
    summary_path = case_dir / f"{channel}_snr_{snr_tag}_{estimator}_run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    ce = summary["channel_estimation_quality"]
    csi = summary["csi_feedback_quality"]
    return {
        "estimator": estimator,
        "channel": channel,
        "snr_db": snr,
        "doppler_hz": doppler,
        "pilot_spacing": pilot_spacing,
        "semantic_psnr_db": summary["reconstructed"].get("psnr_db"),
        "ce_nmse_db": ce.get("h_est_nmse_db"),
        "semantic_evm_db": ce.get("semantic_evm_db"),
        "bs_csi_nmse_db": csi.get("bs_recovered_csi_nmse_db"),
        "bs_true_csi_nmse_db": csi.get("bs_recovered_true_csi_nmse_db"),
        "estimator_runtime_ms": ce.get("estimator_inference_time_ms"),
        "feedback_method": csi.get("feedback_method"),
        "summary_path": str(summary_path),
    }


def write_outputs(rows: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "metrics.csv"
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    gain_rows = compute_gain_rows(rows)
    gain_csv_path = output_dir / "gain_vs_baselines.csv"
    if gain_rows:
        gain_fields = sorted({key for row in gain_rows for key in row.keys()})
        with gain_csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=gain_fields)
            writer.writeheader()
            writer.writerows(gain_rows)

    lines = ["# Comm Foundation Grid Evaluation", ""]
    lines.append(f"- Cases: `{len(rows)}`")
    lines.append(f"- Metrics CSV: `{csv_path}`")
    if gain_rows:
        lines.append(f"- Gain CSV: `{gain_csv_path}`")
    lines.append("")
    lines.append("| Estimator | Channel | SNR | Doppler | Pilot | PSNR | CE NMSE | EVM | BS CSI NMSE | Runtime |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        if "command" in row:
            continue
        lines.append(
            f"| {row['estimator']} | {row['channel']} | {row['snr_db']} | {row['doppler_hz']} | "
            f"{row['pilot_spacing']} | {row['semantic_psnr_db']:.2f} | {row['ce_nmse_db']:.2f} | "
            f"{row['semantic_evm_db']:.2f} | {row['bs_csi_nmse_db']:.2f} | {row['estimator_runtime_ms']:.3f} |"
        )
    if gain_rows:
        lines.extend(["", "## Gain Statistics", ""])
        lines.append("| Estimator | Baseline | Mean CE Gain | Mean PSNR Gain | Mean EVM Gain | Mean Runtime Delta |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for row in summarize_gains(gain_rows):
            lines.append(
                f"| {row['estimator']} | {row['baseline']} | {row['mean_ce_nmse_gain_db']:.3f} | "
                f"{row['mean_psnr_gain_db']:.3f} | {row['mean_evm_gain_db']:.3f} | "
                f"{row['mean_runtime_delta_ms']:.3f} |"
            )
        lines.extend(
            [
                "",
                "## Key Questions",
                "",
                "- `comm_foundation_trained` 相对 `ls_smoothing` 的平均收益见上表；如果 CE/PSNR/EVM 增益接近 0，说明当前 residual 还没有超过结构先验。",
                "- `comm_foundation_trained` 相对 `comm_foundation_untrained` 的收益可用于判断 checkpoint 是否学到了非零修正。",
                "- CE/EVM/BS CSI gain 采用 baseline - estimator，因此正数表示 NMSE/EVM 更低；PSNR gain 采用 estimator - baseline，正数表示图像质量更高。",
                "- runtime delta 用于判断 trained residual 的推理开销是否值得。"
            ]
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def case_key(row: dict) -> tuple:
    return (row["channel"], row["snr_db"], row["doppler_hz"], row["pilot_spacing"])


def compute_gain_rows(rows: list[dict]) -> list[dict]:
    valid = [row for row in rows if "command" not in row]
    by_estimator = {(case_key(row), row["estimator"]): row for row in valid}
    gain_rows = []
    for row in valid:
        if row["estimator"] in {"ls", "ls_smoothing"}:
            continue
        for baseline in ("ls_smoothing", "comm_foundation_untrained"):
            base = by_estimator.get((case_key(row), baseline))
            if base is None or base is row:
                continue
            gain_rows.append(
                {
                    "estimator": row["estimator"],
                    "baseline": baseline,
                    "channel": row["channel"],
                    "snr_db": row["snr_db"],
                    "doppler_hz": row["doppler_hz"],
                    "pilot_spacing": row["pilot_spacing"],
                    "ce_nmse_gain_db": base["ce_nmse_db"] - row["ce_nmse_db"],
                    "psnr_gain_db": row["semantic_psnr_db"] - base["semantic_psnr_db"],
                    "evm_gain_db": base["semantic_evm_db"] - row["semantic_evm_db"],
                    "bs_csi_gain_db": base["bs_csi_nmse_db"] - row["bs_csi_nmse_db"],
                    "runtime_delta_ms": row["estimator_runtime_ms"] - base["estimator_runtime_ms"],
                }
            )
    return gain_rows


def summarize_gains(gain_rows: list[dict]) -> list[dict]:
    groups = {}
    for row in gain_rows:
        groups.setdefault((row["estimator"], row["baseline"]), []).append(row)
    summaries = []
    for (estimator, baseline), group in sorted(groups.items()):
        summaries.append(
            {
                "estimator": estimator,
                "baseline": baseline,
                "mean_ce_nmse_gain_db": sum(row["ce_nmse_gain_db"] for row in group) / len(group),
                "mean_psnr_gain_db": sum(row["psnr_gain_db"] for row in group) / len(group),
                "mean_evm_gain_db": sum(row["evm_gain_db"] for row in group) / len(group),
                "mean_runtime_delta_ms": sum(row["runtime_delta_ms"] for row in group) / len(group),
            }
        )
    return summaries


def main() -> None:
    args = build_arg_parser().parse_args()
    estimators = parse_list(args.estimators, str)
    snrs = parse_list(args.snrs, float)
    dopplers = parse_list(args.dopplers, float)
    channels = parse_list(args.channels, str)
    pilot_spacings = parse_list(args.pilot_spacings, int)

    rows = []
    for estimator in estimators:
        for channel in channels:
            for snr in snrs:
                for doppler in dopplers:
                    for pilot_spacing in pilot_spacings:
                        rows.append(run_case(args, estimator, channel, snr, doppler, pilot_spacing))
    write_outputs(rows, Path(args.output_dir))
    print(json.dumps({"cases": len(rows), "output_dir": args.output_dir}, indent=2))


if __name__ == "__main__":
    main()
