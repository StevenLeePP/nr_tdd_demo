from __future__ import annotations

import argparse
import contextlib
import json
import sys
import warnings
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from nr_tdd_semantic.config import ChannelConfig, ConventionalConfig, DemoConfig, NRPhyConfig, SemanticConfig
    from nr_tdd_semantic.simulation import TDDPhysicalLayerSimulation
    from nr_tdd_semantic.visualization import build_output_paths
else:
    from .config import ChannelConfig, ConventionalConfig, DemoConfig, NRPhyConfig, SemanticConfig
    from .simulation import TDDPhysicalLayerSimulation
    from .visualization import build_output_paths


def parse_int_list(text: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in text.split(",") if item.strip())


def parse_float_list(text: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in text.split(",") if item.strip())


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the modular NR TDD SwinJSCC PHY demo.")
    parser.add_argument("--image", default=DemoConfig().image_path, help="Input image path.")
    parser.add_argument("--scs-khz", type=int, default=15, help="Subcarrier spacing in kHz.")
    parser.add_argument("--n-fft", type=int, default=1024, help="OFDM FFT size.")
    parser.add_argument("--n-subcarriers", type=int, default=600, help="Number of active subcarriers.")
    parser.add_argument("--dl-slots", type=int, default=72, help="Number of downlink slots.")
    parser.add_argument("--ul-slots", type=int, default=2, help="Number of uplink slots.")
    parser.add_argument("--snr-db", type=float, default=20.0, help="Target AWGN SNR in dB.")
    parser.add_argument(
        "--channel",
        choices=("awgn", "rayleigh", "rician"),
        default="rayleigh",
        help="Channel model.",
    )
    parser.add_argument("--delays", default="0,2,5", help="Comma-separated tap delays in samples.")
    parser.add_argument("--powers-db", default="0,-3,-8", help="Comma-separated tap powers in dB.")
    parser.add_argument("--rician-k-db", type=float, default=6.0, help="Rician K-factor in dB.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument(
        "--semantic",
        choices=("swinjscc", "fallback"),
        default="swinjscc",
        help="Use the real SwinJSCC model or deterministic fallback symbols.",
    )
    parser.add_argument("--h264-crf", type=int, default=28, help="H.264 CRF for the conventional branch.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    phy_cfg = NRPhyConfig(
        scs_khz=args.scs_khz,
        n_fft=args.n_fft,
        n_subcarriers=args.n_subcarriers,
        n_dl_slots=args.dl_slots,
        n_ul_slots=args.ul_slots,
        snr_db=args.snr_db,
        rng_seed=args.seed,
    )
    channel_cfg = ChannelConfig(
        channel_type=args.channel,
        delays=parse_int_list(args.delays),
        powers_db=parse_float_list(args.powers_db),
        rician_k_db=args.rician_k_db,
    )
    semantic_cfg = SemanticConfig(use_real_swinjscc=args.semantic == "swinjscc")
    conventional_cfg = ConventionalConfig(h264_crf=args.h264_crf)
    demo_cfg = DemoConfig(image_path=args.image)

    output_paths = build_output_paths(demo_cfg.output_dir, channel_cfg.channel_type, phy_cfg.snr_db)
    raw_console_path = Path(output_paths["raw_console"])
    raw_console_path.parent.mkdir(parents=True, exist_ok=True)

    with raw_console_path.open("w", encoding="utf-8") as raw_console:
        with contextlib.redirect_stdout(raw_console), contextlib.redirect_stderr(raw_console):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sim = TDDPhysicalLayerSimulation(
                    phy_cfg, channel_cfg, semantic_cfg, conventional_cfg, demo_cfg
                )
                result = sim.run()

    write_logs(result.summary(), result.output_paths)
    key = result.summary()
    semantic_psnr = key["reconstructed"].get("psnr_db")
    traditional_psnr = key["conventional"].get("psnr_db")
    csi_nmse_db = key["csi_feedback_quality"].get("bs_recovered_csi_nmse_db")
    print(
        "Done. "
        f"Channel={key['channel_type']}, SNR={key['target_snr_db']} dB, "
        f"SwinJSCC PSNR={semantic_psnr:.2f} dB, "
        f"H.264+LDPC PSNR={traditional_psnr:.2f} dB, "
        f"BS CSI NMSE={csi_nmse_db:.2f} dB."
    )
    print(f"Summary: {result.output_paths['run_summary_json']}")
    print(f"Artifacts: {demo_cfg.output_dir}")


def write_logs(summary: dict, output_paths: dict[str, str]) -> None:
    json_path = Path(output_paths["run_summary_json"])
    md_path = Path(output_paths["run_summary_md"])
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    conventional = summary["conventional"]
    reconstructed = summary["reconstructed"]
    csi = summary["csi_feedback_quality"]
    lines = [
        "# NR TDD Semantic PHY Run Summary",
        "",
        f"- Channel: `{summary['channel_type']}`",
        f"- Target SNR: `{summary['target_snr_db']} dB`",
        f"- DL measured SNR: `{summary['dl_measured_snr_db']:.2f} dB`",
        f"- UL measured SNR: `{summary['ul_measured_snr_db']:.2f} dB`",
        f"- DL used symbols: `{summary['dl_used_symbols']}`",
        "",
        "## Image Reconstruction",
        "",
        f"- SwinJSCC PSNR: `{reconstructed.get('psnr_db'):.2f} dB`",
        f"- H.264+LDPC PSNR: `{conventional.get('psnr_db'):.2f} dB`",
        f"- H.264 bytes: `{conventional.get('h264_bytes')}`",
        f"- H.264 CRF: `{conventional.get('h264_crf')}`",
        f"- LDPC rate: `{conventional.get('ldpc_rate')}`",
        f"- LDPC coded bits: `{conventional.get('ldpc_coded_bits')}`",
        f"- Transmitted bits after repetition: `{conventional.get('transmitted_bits')}`",
        f"- Repetition factor: `{conventional.get('repetition_factor')}`",
        f"- LDPC input bit errors after repetition: `{conventional.get('ldpc_input_bit_errors_after_repetition')}`",
        f"- Decoded payload bit errors: `{conventional.get('decoded_payload_bit_errors')}`",
        "",
        "## CSI Feedback",
        "",
        f"- Feedback bits: `{csi.get('feedback_bits')}`",
        f"- Feedback BER: `{csi.get('feedback_ber'):.6g}`",
        f"- UE compression NMSE: `{csi.get('ue_compression_nmse_db'):.2f} dB`",
        f"- BS recovered CSI NMSE: `{csi.get('bs_recovered_csi_nmse_db'):.2f} dB`",
        "",
        "## Artifacts",
        "",
    ]
    for name, path in output_paths.items():
        lines.append(f"- {name}: `{path}`")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
