#!/usr/bin/env python3
"""Read experiment artifacts and recommend the next research action."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_json(path: str | None):
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def read_csv(path: str | None) -> list[dict]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(row: dict, key: str, default: float = float("nan")) -> float:
    try:
        value = row.get(key, default)
        if value == "":
            return default
        return float(value)
    except Exception:
        return default


def mean(values: list[float]) -> float:
    vals = [v for v in values if v == v]
    return sum(vals) / len(vals) if vals else float("nan")


def decide(args) -> tuple[str, list[str], dict]:
    diagnostics = read_json(args.checkpoint_diagnostics)
    overfit = read_json(args.overfit_debug)
    training_rows = read_csv(args.training_results)
    gain_rows = read_csv(args.gain_vs_baselines)
    reliability_rows = read_csv(args.reliability_metrics)
    e2e_rows = read_csv(args.end_to_end_metrics)
    evidence = {}
    reasons = []

    residual_norm = None
    trained_untrained_delta = None
    if diagnostics:
        residual_norm = diagnostics.get("residual_stats_trained", {}).get("l2_norm")
        trained_untrained_delta = abs(
            float(diagnostics.get("comm_foundation_trained_nmse", 0.0))
            - float(diagnostics.get("comm_foundation_untrained_nmse", 0.0))
        )
        evidence["diagnostic_residual_l2"] = residual_norm
        evidence["trained_untrained_nmse_delta"] = trained_untrained_delta
        if residual_norm is not None and residual_norm <= args.zero_threshold:
            reasons.append("checkpoint diagnostic shows residual norm is approximately zero")
            return "debug_residual_training", reasons, evidence
        if trained_untrained_delta <= args.zero_threshold:
            reasons.append("trained and untrained checkpoint outputs are effectively identical")
            return "debug_residual_training", reasons, evidence

    if overfit:
        evidence["overfit_can_overfit"] = bool(overfit.get("can_overfit", False))
        evidence["overfit_residual_norm_after"] = overfit.get("residual_norm_after")
        evidence["overfit_gain_nmse"] = overfit.get("h_hat_gain_over_structured_nmse")
        if not overfit.get("can_overfit", False):
            reasons.append("tiny overfit debug did not prove residual trainability")
            return "debug_residual_training", reasons, evidence

    trained_vs_smoothing = [
        row for row in gain_rows
        if row.get("estimator") == "comm_foundation_trained" and row.get("baseline") == "ls_smoothing"
    ]
    trained_vs_untrained = [
        row for row in gain_rows
        if row.get("estimator") == "comm_foundation_trained" and row.get("baseline") == "comm_foundation_untrained"
    ]
    ce_gain = mean([as_float(row, "ce_nmse_gain_db") for row in trained_vs_smoothing])
    psnr_gain = mean([as_float(row, "psnr_gain_db") for row in trained_vs_smoothing])
    untrained_ce_gain = mean([as_float(row, "ce_nmse_gain_db") for row in trained_vs_untrained])
    evidence["mean_trained_vs_ls_smoothing_ce_gain_db"] = ce_gain
    evidence["mean_trained_vs_ls_smoothing_psnr_gain_db"] = psnr_gain
    evidence["mean_trained_vs_untrained_ce_gain_db"] = untrained_ce_gain

    if trained_vs_smoothing and (ce_gain <= 0.0 and psnr_gain <= 0.0):
        reasons.append("trained residual does not beat ls_smoothing on available gain rows")
        return "do_not_expand_channel_estimator", reasons, evidence

    if e2e_rows:
        ce_vals = [as_float(row, "ce_nmse_db") for row in e2e_rows if row.get("estimator") != "ls"]
        csi_vals = [as_float(row, "bs_csi_nmse_db") for row in e2e_rows if row.get("estimator") != "ls"]
        if ce_vals and csi_vals:
            evidence["mean_non_ls_ce_nmse_db"] = mean(ce_vals)
            evidence["mean_non_ls_bs_csi_nmse_db"] = mean(csi_vals)
            if mean(ce_vals) < args.good_ce_db and mean(csi_vals) > args.poor_csi_db:
                reasons.append("UE-side CE NMSE is strong while BS CSI feedback remains weak")
                return "prioritize_learned_csi_feedback", reasons, evidence

    if reliability_rows:
        ratios = [as_float(row, "high_low_error_ratio") for row in reliability_rows]
        corr = [as_float(row, "reliability_error_corr") for row in reliability_rows]
        evidence["mean_high_low_error_ratio"] = mean(ratios)
        evidence["mean_reliability_error_corr"] = mean(corr)
        if ratios and mean(ratios) < args.reliability_ratio_threshold:
            reasons.append("ReliabilityHead separates high and low reliability error regions")
            return "try_reliability_guided_mapping", reasons, evidence

    if training_rows:
        pretrain_rows = [row for row in training_rows if row.get("training_strategy") in {"pretrain_then_finetune", "joint_pretrain_ce"}]
        scratch_rows = [row for row in training_rows if row.get("training_strategy") == "scratch_ce"]
        improvements = []
        for pre in pretrain_rows:
            base = next((row for row in scratch_rows if row.get("sample_fraction") == pre.get("sample_fraction")), None)
            if base:
                improvements.append(as_float(base, "val_nmse_db") - as_float(pre, "val_nmse_db"))
        evidence["pretrain_vs_scratch_val_gain_db"] = mean(improvements)
        if improvements and mean(improvements) >= args.pretrain_gain_threshold:
            reasons.append("pretraining beats scratch under available few-shot comparisons")
            if ce_gain >= 1.0 and psnr_gain >= 0.1:
                return "consider_complex_transformer", reasons, evidence
            return "continue_foundation_pretraining", reasons, evidence

    reasons.append("available evidence does not justify neural expansion; continue residual/debug-controlled experiments")
    return "continue_residual_debug", reasons, evidence


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_diagnostics", default="outputs/checkpoint_diagnostics/checkpoint_diagnostics.json")
    parser.add_argument("--overfit_debug", default="outputs/residual_overfit_debug/overfit_residual_debug.json")
    parser.add_argument("--training_results", default="outputs/training_comparison/training_results.csv")
    parser.add_argument("--gain_vs_baselines", default="outputs/grid_eval/gain_vs_baselines.csv")
    parser.add_argument("--reliability_metrics", default="outputs/reliability_eval/reliability_metrics.csv")
    parser.add_argument("--end_to_end_metrics", default="outputs/end_to_end_retest/end_to_end_metrics.csv")
    parser.add_argument("--output_dir", default="outputs/adaptive_decision")
    parser.add_argument("--zero_threshold", type=float, default=1e-8)
    parser.add_argument("--good_ce_db", type=float, default=-25.0)
    parser.add_argument("--poor_csi_db", type=float, default=-10.0)
    parser.add_argument("--reliability_ratio_threshold", type=float, default=0.8)
    parser.add_argument("--pretrain_gain_threshold", type=float, default=0.5)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    next_action, reasons, evidence = decide(args)
    payload = {
        "next_action": next_action,
        "reasons": reasons,
        "evidence": evidence,
    }
    (output_dir / "next_action.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Adaptive Experiment Decision",
        "",
        f"- Next action: `{next_action}`",
        "",
        "## Reasons",
        "",
    ]
    lines.extend([f"- {reason}" for reason in reasons])
    lines.extend(["", "## Evidence", ""])
    for key, value in evidence.items():
        lines.append(f"- `{key}`: `{value}`")
    (output_dir / "decision_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
