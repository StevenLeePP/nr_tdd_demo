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
    parser.add_argument("--output-dir", default=str(DemoConfig().output_dir), help="Directory for run artifacts.")
    parser.add_argument("--scs-khz", type=int, default=15, help="Subcarrier spacing in kHz.")
    parser.add_argument("--n-fft", type=int, default=1024, help="OFDM FFT size.")
    parser.add_argument("--n-subcarriers", type=int, default=600, help="Number of active subcarriers.")
    parser.add_argument("--dl-slots", type=int, default=72, help="Number of downlink slots.")
    parser.add_argument("--ul-slots", type=int, default=2, help="Number of uplink slots.")
    parser.add_argument("--pilot-spacing", type=int, default=4, help="Comb pilot spacing in subcarriers.")
    parser.add_argument("--num-tx-antennas", type=int, default=1, help="Number of BS transmit antennas.")
    parser.add_argument("--num-rx-antennas", type=int, default=1, help="Number of UE receive antennas.")
    parser.add_argument("--array-type", default="ula", choices=("ula",), help="Antenna array type.")
    parser.add_argument("--array-size", default="1x1", help="Linear array size, e.g. 1x4.")
    parser.add_argument("--ul-num-tx-antennas", type=int, default=1, help="Number of UE transmit antennas for UL feedback.")
    parser.add_argument("--ul-num-rx-antennas", type=int, default=1, help="Number of BS receive antennas for UL feedback.")
    parser.add_argument("--ul-array-size", default="1x1", help="UL linear array size, e.g. 1x4.")
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
    parser.add_argument("--doppler-hz", type=float, default=0.0, help="Maximum per-path Doppler shift in Hz.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument(
        "--semantic",
        choices=("swinjscc", "fallback"),
        default="swinjscc",
        help="Use the real SwinJSCC model or deterministic fallback symbols.",
    )
    parser.add_argument("--h264-crf", type=int, default=28, help="H.264 CRF for the conventional branch.")
    parser.add_argument(
        "--channel-estimator",
        choices=("ls", "ls_smoothing", "comm_foundation_untrained", "comm_foundation_trained", "comm_foundation"),
        default="ls",
        help="Channel estimator used at the UE equalizer.",
    )
    parser.add_argument(
        "--comm-foundation-checkpoint",
        default=None,
        help="Checkpoint path for --channel-estimator comm_foundation.",
    )
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
        pilot_spacing=args.pilot_spacing,
        num_tx_antennas=args.num_tx_antennas,
        num_rx_antennas=args.num_rx_antennas,
        array_type=args.array_type,
        array_size=args.array_size,
        ul_num_tx_antennas=args.ul_num_tx_antennas,
        ul_num_rx_antennas=args.ul_num_rx_antennas,
        ul_array_size=args.ul_array_size,
    )
    channel_cfg = ChannelConfig(
        channel_type=args.channel,
        delays=parse_int_list(args.delays),
        powers_db=parse_float_list(args.powers_db),
        rician_k_db=args.rician_k_db,
        doppler_hz=args.doppler_hz,
    )
    semantic_cfg = SemanticConfig(use_real_swinjscc=args.semantic == "swinjscc")
    conventional_cfg = ConventionalConfig(h264_crf=args.h264_crf)
    demo_cfg = DemoConfig(image_path=args.image, output_dir=Path(args.output_dir))

    output_paths = build_output_paths(
        demo_cfg.output_dir,
        channel_cfg.channel_type,
        phy_cfg.snr_db,
        estimator_tag=args.channel_estimator,
    )
    raw_console_path = Path(output_paths["raw_console"])
    raw_console_path.parent.mkdir(parents=True, exist_ok=True)

    with raw_console_path.open("w", encoding="utf-8") as raw_console:
        with contextlib.redirect_stdout(raw_console), contextlib.redirect_stderr(raw_console):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sim = TDDPhysicalLayerSimulation(
                    phy_cfg,
                    channel_cfg,
                    semantic_cfg,
                    conventional_cfg,
                    demo_cfg,
                    channel_estimator=args.channel_estimator,
                    comm_foundation_checkpoint=args.comm_foundation_checkpoint,
                )
                result = sim.run()

    write_logs(result.summary(), result.output_paths)
    key = result.summary()
    semantic_psnr = key["reconstructed"].get("psnr_db")
    traditional_psnr = key["conventional"].get("psnr_db")
    csi_nmse_db = key["csi_feedback_quality"].get("bs_recovered_csi_nmse_db")
    ce_nmse_db = key["channel_estimation_quality"].get("h_est_nmse_db")
    print(
        "Done. "
        f"Channel={key['channel_type']}, SNR={key['target_snr_db']} dB, "
        f"SwinJSCC PSNR={format_metric(semantic_psnr, suffix=' dB')}, "
        f"H.264+LDPC PSNR={format_metric(traditional_psnr, suffix=' dB')}, "
        f"CE NMSE={ce_nmse_db:.2f} dB, "
        f"BS CSI NMSE={csi_nmse_db:.2f} dB, "
        f"Feedback={key['csi_feedback_quality'].get('feedback_method')}, "
        f"H_true_shape={key['channel_estimation_quality'].get('h_true_shape')}, "
        f"H_est_shape={key['channel_estimation_quality'].get('h_est_shape')}."
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
    ce = summary["channel_estimation_quality"]
    lines = [
        "# NR TDD Semantic PHY Run Summary",
        "",
        f"- Channel: `{summary['channel_type']}`",
        f"- Target SNR: `{summary['target_snr_db']} dB`",
        f"- DL measured SNR: `{summary['dl_measured_snr_db']:.2f} dB`",
        f"- UL measured SNR: `{summary['ul_measured_snr_db']:.2f} dB`",
        f"- Doppler: `{summary['channel_doppler_hz']} Hz`",
        f"- DL Tx/Rx antennas: `{summary.get('num_tx_antennas', 1)} x {summary.get('num_rx_antennas', 1)}`",
        f"- UL Tx/Rx antennas: `{summary.get('ul_num_tx_antennas', 1)} x {summary.get('ul_num_rx_antennas', 1)}`",
        f"- DL Array: `{summary.get('array_type', 'ula')} {summary.get('array_size', '1x1')}`",
        f"- UL Array: `{summary.get('array_type', 'ula')} {summary.get('ul_array_size', '1x1')}`",
        f"- DL used symbols: `{summary['dl_used_symbols']}`",
        "",
        "## Image Reconstruction",
        "",
        f"- SwinJSCC PSNR: `{format_metric(reconstructed.get('psnr_db'), suffix=' dB')}`",
        f"- H.264+LDPC PSNR: `{format_metric(conventional.get('psnr_db'), suffix=' dB')}`",
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
        f"- Feedback method: `{csi.get('feedback_method')}`",
        f"- Feedback bits: `{csi.get('feedback_bits')}`",
        f"- Feedback BER: `{format_metric(csi.get('feedback_ber'), precision=6)}`",
        f"- UE compression NMSE: `{format_metric(csi.get('ue_compression_nmse_db'), suffix=' dB')}`",
        f"- BS recovered CSI NMSE: `{format_metric(csi.get('bs_recovered_csi_nmse_db'), suffix=' dB')}`",
        f"- BS recovered true CSI NMSE: `{format_metric(csi.get('bs_recovered_true_csi_nmse_db'), suffix=' dB')}`",
        f"- UL X_tx shape: `{csi.get('ul_x_tx_shape')}`",
        f"- UL Y_rx shape: `{csi.get('ul_y_rx_shape')}`",
        f"- UL H_true shape: `{csi.get('ul_h_true_shape')}`",
        "",
        "## Channel Estimation",
        "",
        f"- Estimator: `{ce.get('estimator')}`",
        f"- H_hat NMSE: `{ce.get('h_est_nmse_db'):.2f} dB`",
        f"- Semantic equalized EVM: `{ce.get('semantic_evm_db'):.2f} dB`",
        f"- Estimator inference time: `{ce.get('estimator_inference_time_ms'):.3f} ms`",
        f"- X_tx shape: `{ce.get('x_tx_shape')}`",
        f"- Y_rx shape: `{ce.get('y_rx_shape')}`",
        f"- H_true shape: `{ce.get('h_true_shape')}`",
        f"- H_est shape: `{ce.get('h_est_shape')}`",
        "",
        "## Artifacts",
        "",
    ]
    for name, path in output_paths.items():
        lines.append(f"- {name}: `{path}`")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_metric(value, precision: int = 2, suffix: str = "") -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.{precision}f}{suffix}"
    except (TypeError, ValueError):
        return str(value)


if __name__ == "__main__":
    main()
