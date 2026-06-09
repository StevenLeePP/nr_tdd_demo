#!/usr/bin/env python3
"""Stress-test channel estimators under low SNR and nonideal channels."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PACKAGE_DIR = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = PACKAGE_DIR.parent
sys.path.insert(0, str(PACKAGE_PARENT))

from nr_tdd_semantic.config import NRPhyConfig  # noqa: E402
from nr_tdd_semantic.dsp import csi_nmse, delay_domain_denoise_csi  # noqa: E402
from nr_tdd_semantic.models.comm_foundation_model import (  # noqa: E402
    CommFoundationChannelEstimator,
    CommFoundationConfig,
    channels_to_complex_np,
    complex_np_to_channels,
)
from nr_tdd_semantic.scripts.export_comm_foundation_dataset import generate_sample  # noqa: E402


def parse_list(text: str, cast):
    return [cast(item.strip()) for item in text.split(",") if item.strip()]


def linear_to_db(value: float) -> float:
    return 10.0 * np.log10(max(float(value), 1e-30))


def random_delay_profile(rng: np.random.Generator, mode: str) -> tuple[tuple[int, ...], tuple[float, ...]]:
    if mode == "default":
        return (0, 2, 5), (0.0, -3.0, -8.0)
    if mode == "extended":
        delay1 = int(rng.integers(3, 8))
        delay2 = int(rng.integers(max(delay1 + 1, 9), 22))
        powers = (0.0, -float(rng.uniform(4.0, 8.0)), -float(rng.uniform(10.0, 18.0)))
        return (0, delay1, delay2), powers
    if mode == "mixed":
        return random_delay_profile(rng, "extended" if rng.random() < 0.5 else "default")
    raise ValueError("--delay-profile must be default, extended, or mixed.")


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


def structured_estimate(h_ls: np.ndarray, cfg: NRPhyConfig, doppler_hz: float, args) -> np.ndarray:
    if args.time_average == "always":
        time_average = True
    elif args.time_average == "never":
        time_average = False
    else:
        time_average = float(doppler_hz) <= args.time_average_doppler_threshold
    return delay_domain_denoise_csi(
        h_ls,
        cfg,
        n_taps=args.delay_denoise_taps,
        time_average=time_average,
    ).astype(np.complex64)


@torch.no_grad()
def neural_estimate(
    model: CommFoundationChannelEstimator,
    h_structured: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    x = complex_np_to_channels(h_structured.astype(np.complex64)).to(device)
    pred = model(x)
    return channels_to_complex_np(pred).astype(np.complex64)


def evaluate_one(
    estimator: str,
    h_ls: np.ndarray,
    h_true: np.ndarray,
    cfg: NRPhyConfig,
    doppler_hz: float,
    args,
    models: dict[str, CommFoundationChannelEstimator],
    device: torch.device,
) -> tuple[float, float]:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    if estimator == "ls":
        h_hat = h_ls
    elif estimator == "ls_smoothing":
        h_hat = structured_estimate(h_ls, cfg, doppler_hz, args)
    elif estimator in {"comm_foundation_untrained", "comm_foundation_trained"}:
        h_structured = structured_estimate(h_ls, cfg, doppler_hz, args)
        h_hat = neural_estimate(models[estimator], h_structured, device)
    else:
        raise ValueError(f"Unknown estimator: {estimator}")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    runtime_ms = (time.perf_counter() - start) * 1000.0
    return csi_nmse(h_true, h_hat), runtime_ms


def compute_gain_rows(rows: list[dict]) -> list[dict]:
    by_case = {
        (row["channel"], row["snr_db"], row["doppler_hz"], row["pilot_spacing"], row["estimator"]): row
        for row in rows
    }
    gain_rows = []
    for row in rows:
        if row["estimator"] == "ls":
            continue
        baselines = ["ls"]
        if row["estimator"] not in {"ls_smoothing"}:
            baselines.extend(["ls_smoothing", "comm_foundation_untrained"])
        for baseline in baselines:
            if baseline == row["estimator"]:
                continue
            base = by_case.get((row["channel"], row["snr_db"], row["doppler_hz"], row["pilot_spacing"], baseline))
            if base is None:
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
                    "runtime_delta_ms": row["runtime_ms"] - base["runtime_ms"],
                }
            )
    return gain_rows


def summarize_gains(gain_rows: list[dict]) -> list[dict]:
    groups = {}
    for row in gain_rows:
        groups.setdefault((row["estimator"], row["baseline"]), []).append(row)
    summary = []
    for (estimator, baseline), group in sorted(groups.items()):
        gains = [row["ce_nmse_gain_db"] for row in group]
        summary.append(
            {
                "estimator": estimator,
                "baseline": baseline,
                "cases": len(group),
                "mean_ce_gain_db": float(np.mean(gains)),
                "median_ce_gain_db": float(np.median(gains)),
                "positive_gain_ratio": float(np.mean(np.asarray(gains) > 0.0)),
                "mean_runtime_delta_ms": float(np.mean([row["runtime_delta_ms"] for row in group])),
            }
        )
    return summary


def write_outputs(rows: list[dict], sample_rows: list[dict], output_dir: Path, args) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.csv"
    sample_path = output_dir / "sample_metrics.csv"
    gain_path = output_dir / "gain_vs_baselines.csv"
    for path, data in ((metrics_path, rows), (sample_path, sample_rows)):
        fieldnames = sorted({key for row in data for key in row.keys()})
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

    gain_rows = compute_gain_rows(rows)
    if gain_rows:
        fieldnames = sorted({key for row in gain_rows for key in row.keys()})
        with gain_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(gain_rows)

    lines = [
        "# Low-SNR Doppler Channel Estimator Stress Test",
        "",
        f"- SNR sweep: `{args.snrs}`",
        f"- Doppler sweep: `{args.dopplers}`",
        f"- Channels: `{args.channels}`",
        f"- Pilot spacings: `{args.pilot_spacings}`",
        f"- Samples per case: `{args.samples_per_case}`",
        f"- Delay profile: `{args.delay_profile}`",
        f"- Delay denoise taps: `{args.delay_denoise_taps}`",
        f"- Metrics CSV: `{metrics_path}`",
        f"- Sample CSV: `{sample_path}`",
    ]
    if gain_rows:
        lines.append(f"- Gain CSV: `{gain_path}`")
    lines.extend(["", "## Metrics", ""])
    lines.append("| Estimator | Channel | SNR | Doppler | Pilot | NMSE dB | Runtime ms |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            f"| {row['estimator']} | {row['channel']} | {row['snr_db']} | {row['doppler_hz']} | "
            f"{row['pilot_spacing']} | {row['ce_nmse_db']:.2f} | {row['runtime_ms']:.3f} |"
        )
    if gain_rows:
        lines.extend(["", "## Gain Summary", ""])
        lines.append("| Estimator | Baseline | Cases | Mean CE Gain dB | Median CE Gain dB | Positive Ratio | Runtime Delta ms |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for row in summarize_gains(gain_rows):
            lines.append(
                f"| {row['estimator']} | {row['baseline']} | {row['cases']} | "
                f"{row['mean_ce_gain_db']:.3f} | {row['median_ce_gain_db']:.3f} | "
                f"{row['positive_gain_ratio']:.2f} | {row['mean_runtime_delta_ms']:.3f} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation Rule",
            "",
            "- CE gain uses `baseline_nmse_db - estimator_nmse_db`; positive means the estimator has lower NMSE.",
            "- ML only counts as stronger when `comm_foundation_trained` is positive against both `ls_smoothing` and `comm_foundation_untrained`, not merely against raw LS.",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimators", default="ls,ls_smoothing,comm_foundation_untrained,comm_foundation_trained")
    parser.add_argument("--snrs", default="-10,-5,0,5,10")
    parser.add_argument("--dopplers", default="0,30,60,100,150")
    parser.add_argument("--channels", default="rayleigh,rician")
    parser.add_argument("--pilot-spacings", default="4,6,8,12")
    parser.add_argument("--samples-per-case", type=int, default=8)
    parser.add_argument("--checkpoint", default="outputs/comm_foundation_ckpt_residual_safe/best_comm_foundation_channel_estimator.pt")
    parser.add_argument("--output-dir", default="outputs/low_snr_doppler_ce_stress")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--n-fft", type=int, default=128)
    parser.add_argument("--n-subcarriers", type=int, default=72)
    parser.add_argument("--dl-slots", type=int, default=1)
    parser.add_argument("--delay-profile", choices=("default", "extended", "mixed"), default="mixed")
    parser.add_argument("--rician-k-min", type=float, default=-3.0)
    parser.add_argument("--rician-k-max", type=float, default=12.0)
    parser.add_argument("--delay-denoise-taps", type=int, default=16)
    parser.add_argument("--time-average", choices=("auto", "always", "never"), default="auto")
    parser.add_argument("--time-average-doppler-threshold", type=float, default=1e-9)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    estimators = parse_list(args.estimators, str)
    snrs = parse_list(args.snrs, float)
    dopplers = parse_list(args.dopplers, float)
    channels = parse_list(args.channels, str)
    pilot_spacings = parse_list(args.pilot_spacings, int)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    models = {
        "comm_foundation_untrained": load_model(None, device),
    }
    if "comm_foundation_trained" in estimators:
        models["comm_foundation_trained"] = load_model(args.checkpoint, device)

    rows = []
    sample_rows = []
    for channel in channels:
        for snr in snrs:
            for doppler in dopplers:
                for pilot_spacing in pilot_spacings:
                    cfg = NRPhyConfig(
                        n_fft=args.n_fft,
                        n_subcarriers=args.n_subcarriers,
                        n_dl_slots=args.dl_slots,
                        n_ul_slots=1,
                        snr_db=snr,
                        rng_seed=int(rng.integers(0, 2**31 - 1)),
                        pilot_spacing=pilot_spacing,
                    )
                    accum = {estimator: {"nmse": [], "runtime_ms": []} for estimator in estimators}
                    for sample_idx in range(args.samples_per_case):
                        delays, powers_db = random_delay_profile(rng, args.delay_profile)
                        rician_k_db = float(rng.uniform(args.rician_k_min, args.rician_k_max))
                        if args.dry_run:
                            continue
                        sample = generate_sample(
                            rng,
                            cfg,
                            channel,
                            doppler,
                            delays=delays,
                            powers_db=powers_db,
                            rician_k_db=rician_k_db,
                        )
                        h_ls = sample["H_ls_grid"]
                        h_true = sample["H_true"]
                        for estimator in estimators:
                            nmse, runtime_ms = evaluate_one(estimator, h_ls, h_true, cfg, doppler, args, models, device)
                            accum[estimator]["nmse"].append(nmse)
                            accum[estimator]["runtime_ms"].append(runtime_ms)
                            sample_rows.append(
                                {
                                    "estimator": estimator,
                                    "channel": channel,
                                    "snr_db": snr,
                                    "doppler_hz": doppler,
                                    "pilot_spacing": pilot_spacing,
                                    "sample_idx": sample_idx,
                                    "ce_nmse": nmse,
                                    "ce_nmse_db": linear_to_db(nmse),
                                    "runtime_ms": runtime_ms,
                                    "delays": ",".join(str(x) for x in delays),
                                    "powers_db": ",".join(f"{x:g}" for x in powers_db),
                                    "rician_k_db": rician_k_db,
                                }
                            )
                    if args.dry_run:
                        continue
                    for estimator in estimators:
                        nmse_values = accum[estimator]["nmse"]
                        runtime_values = accum[estimator]["runtime_ms"]
                        mean_nmse = float(np.mean(nmse_values))
                        rows.append(
                            {
                                "estimator": estimator,
                                "channel": channel,
                                "snr_db": snr,
                                "doppler_hz": doppler,
                                "pilot_spacing": pilot_spacing,
                                "samples": len(nmse_values),
                                "ce_nmse": mean_nmse,
                                "ce_nmse_db": linear_to_db(mean_nmse),
                                "ce_nmse_db_mean_per_sample": float(np.mean([linear_to_db(x) for x in nmse_values])),
                                "runtime_ms": float(np.mean(runtime_values)),
                                "delay_profile": args.delay_profile,
                                "delay_denoise_taps": args.delay_denoise_taps,
                                "time_average": args.time_average,
                                "rician_k_min": args.rician_k_min,
                                "rician_k_max": args.rician_k_max,
                            }
                        )

    output_dir = Path(args.output_dir)
    if args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "estimators": estimators,
            "snrs": snrs,
            "dopplers": dopplers,
            "channels": channels,
            "pilot_spacings": pilot_spacings,
            "samples_per_case": args.samples_per_case,
            "cases": len(snrs) * len(dopplers) * len(channels) * len(pilot_spacings),
        }
        (output_dir / "dry_run.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2))
        return
    write_outputs(rows, sample_rows, output_dir, args)
    print(json.dumps({"rows": len(rows), "sample_rows": len(sample_rows), "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
