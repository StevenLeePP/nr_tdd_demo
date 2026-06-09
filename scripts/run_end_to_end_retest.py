#!/usr/bin/env python3
"""Run the end-to-end semantic-link retest grid and export requested filenames."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parents[1]
GRID_SCRIPT = PACKAGE_DIR / "scripts" / "evaluate_comm_foundation_grid.py"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", default=str(PACKAGE_DIR / "outputs" / "end_to_end_retest"))
    parser.add_argument("--channels", default="rayleigh,rician")
    parser.add_argument("--snrs", default="5,10,20")
    parser.add_argument("--dopplers", default="0,60,100")
    parser.add_argument("--pilot-spacings", default="4,8")
    parser.add_argument("--semantic", choices=("swinjscc", "fallback"), default="swinjscc")
    parser.add_argument("--dl-slots", type=int, default=72)
    parser.add_argument("--ul-slots", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = Path(args.output_dir)
    cmd = [
        sys.executable,
        str(GRID_SCRIPT),
        "--estimators",
        "ls,ls_smoothing,comm_foundation_untrained,comm_foundation_trained",
        "--channels",
        args.channels,
        "--snrs",
        args.snrs,
        "--dopplers",
        args.dopplers,
        "--pilot-spacings",
        args.pilot_spacings,
        "--checkpoint",
        args.checkpoint,
        "--output-dir",
        str(output_dir),
        "--semantic",
        args.semantic,
        "--dl-slots",
        str(args.dl_slots),
        "--ul-slots",
        str(args.ul_slots),
        "--seed",
        str(args.seed),
    ]
    if args.dry_run:
        cmd.append("--dry-run")
    subprocess.run(cmd, cwd=PACKAGE_DIR, check=True)
    if not args.dry_run:
        shutil.copyfile(output_dir / "metrics.csv", output_dir / "end_to_end_metrics.csv")
        shutil.copyfile(output_dir / "summary.md", output_dir / "end_to_end_summary.md")
    print(f"End-to-end retest outputs: {output_dir}")


if __name__ == "__main__":
    main()
