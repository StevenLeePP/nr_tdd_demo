from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .config import NRPhyConfig
from .resource_grid import ResourceGridMapper
from .utils import ComplexArray


try:  # pragma: no cover - backend availability is environment-specific.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
except Exception:  # noqa: BLE001
    plt = None
    ListedColormap = None
    Patch = None


def build_output_paths(
    output_dir: Path,
    channel_type: str,
    snr_db: float,
    estimator_tag: str = "ls",
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    snr_tag = f"{snr_db:g}dB".replace(".", "p")
    prefix = f"{channel_type.lower()}_snr_{snr_tag}_{estimator_tag}"
    return {
        "semantic_constellation": str(output_dir / f"{prefix}_semantic_constellation_equalization.png"),
        "traditional_constellation": str(output_dir / f"{prefix}_h264_ldpc_constellation_equalization.png"),
        "reconstruction": str(output_dir / f"{prefix}_reconstruction_comparison.png"),
        "semantic_reconstructed": str(output_dir / f"{prefix}_semantic_reconstructed.png"),
        "traditional_reconstructed": str(output_dir / f"{prefix}_h264_ldpc_reconstructed.png"),
        "resource_grid": str(output_dir / f"{prefix}_time_frequency_resource_grid.png"),
        "frame_structure": str(output_dir / f"{prefix}_frame_structure.png"),
        "run_summary_json": str(output_dir / f"{prefix}_run_summary.json"),
        "run_summary_md": str(output_dir / f"{prefix}_run_summary.md"),
        "raw_console": str(output_dir / f"{prefix}_raw_console.txt"),
    }


def _ensure_matplotlib() -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required for visualization.")


def _sample_constellation(symbols: ComplexArray, max_points: int = 8000) -> ComplexArray:
    symbols = np.asarray(symbols, dtype=np.complex128).reshape(-1)
    if symbols.size <= max_points:
        return symbols
    idx = np.linspace(0, symbols.size - 1, max_points).astype(int)
    return symbols[idx]


def plot_constellation_comparison(
    before_equalization: ComplexArray,
    after_equalization: ComplexArray,
    output_path: str,
    n_complex: Optional[object] = None,
    title_prefix: str = "",
) -> None:
    _ensure_matplotlib()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if n_complex is not None:
        n_complex_int = int(n_complex)
        before_equalization = before_equalization[:n_complex_int]
        after_equalization = after_equalization[:n_complex_int]

    before = _sample_constellation(before_equalization)
    after = _sample_constellation(after_equalization)
    limit = np.percentile(np.abs(np.concatenate([before, after])), 99.0)
    limit = max(float(limit) * 1.15, 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    prefix = f"{title_prefix}: " if title_prefix else ""
    panels = [
        (axes[0], before, f"{prefix}before equalization"),
        (axes[1], after, f"{prefix}after MMSE equalization"),
    ]
    for ax, values, title in panels:
        ax.scatter(np.real(values), np.imag(values), s=4, alpha=0.28, linewidths=0)
        ax.axhline(0.0, color="0.65", linewidth=0.8)
        ax.axvline(0.0, color="0.65", linewidth=0.8)
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title)
        ax.set_xlabel("In-phase")
        ax.set_ylabel("Quadrature")
        ax.grid(True, alpha=0.25)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_reconstruction_comparison(
    original: np.ndarray,
    semantic_reconstructed: np.ndarray,
    traditional_reconstructed: np.ndarray,
    output_path: str,
    semantic_psnr_db: float,
    traditional_psnr_db: float,
) -> None:
    _ensure_matplotlib()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)
    axes[0].imshow(np.clip(original, 0.0, 1.0))
    axes[0].set_title("Input crop")
    axes[0].axis("off")
    axes[1].imshow(np.clip(semantic_reconstructed, 0.0, 1.0))
    axes[1].set_title(f"SwinJSCC, PSNR {semantic_psnr_db:.2f} dB")
    axes[1].axis("off")
    axes[2].imshow(np.clip(traditional_reconstructed, 0.0, 1.0))
    axes[2].set_title(f"H.264+LDPC, PSNR {traditional_psnr_db:.2f} dB")
    axes[2].axis("off")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_time_frequency_resource_grid(
    cfg: NRPhyConfig,
    dl_grids: Sequence[ComplexArray],
    ul_grids: Sequence[ComplexArray],
    dl_allocations: Optional[Sequence[np.ndarray]],
    output_path: str,
) -> None:
    """Visualize pilot/data placement over subcarriers and OFDM symbols."""
    _ensure_matplotlib()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    columns = cfg.symbols_per_slot * (len(dl_grids) + len(ul_grids))
    grid_view = np.zeros((cfg.n_subcarriers, columns), dtype=int)
    links = [("DL", dl_grids), ("UL", ul_grids)]
    col = 0
    for link_name, grids in links:
        mapper = ResourceGridMapper(cfg, link_name)
        for slot_idx, _ in enumerate(grids):
            pilot_mask = mapper.pilot_mask_for_slot(slot_idx)
            if link_name == "DL" and dl_allocations is not None:
                allocation = dl_allocations[slot_idx]
                window = grid_view[:, col : col + cfg.symbols_per_slot]
                window[allocation == mapper.semantic_label] = 1
                window[allocation == mapper.conventional_label] = 2
            else:
                data_mask = mapper.data_mask_for_slot(slot_idx)
                grid_view[:, col : col + cfg.symbols_per_slot][data_mask] = 1
            grid_view[:, col : col + cfg.symbols_per_slot][pilot_mask] = 3
            col += cfg.symbols_per_slot

    cmap = ListedColormap(["#f4f4f4", "#3b82f6", "#22c55e", "#ef4444"])
    fig, ax = plt.subplots(figsize=(13, 6), constrained_layout=True)
    ax.imshow(grid_view, origin="lower", aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=3)
    ax.set_title("Time-frequency resource grid: semantic, H.264+LDPC, and comb pilot REs", pad=24)
    ax.set_xlabel("OFDM symbol index across TDD frame")
    ax.set_ylabel("Active subcarrier index")
    for slot_boundary in range(cfg.symbols_per_slot, columns, cfg.symbols_per_slot):
        ax.axvline(slot_boundary - 0.5, color="black", linewidth=0.8, alpha=0.5)
    dl_boundary = cfg.symbols_per_slot * len(dl_grids)
    ax.axvline(dl_boundary - 0.5, color="black", linewidth=2.0)
    ax.text(
        dl_boundary / 2,
        1.015,
        "DL slots",
        ha="center",
        transform=ax.get_xaxis_transform(),
        clip_on=False,
    )
    ax.text(
        dl_boundary + (columns - dl_boundary) / 2,
        1.015,
        "UL slots",
        ha="center",
        transform=ax.get_xaxis_transform(),
        clip_on=False,
    )
    ax.legend(
        handles=[
            Patch(facecolor="#3b82f6", label="Semantic data RE"),
            Patch(facecolor="#22c55e", label="H.264+LDPC data RE"),
            Patch(facecolor="#ef4444", label="Comb pilot RE"),
        ],
        loc="upper right",
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_frame_structure(cfg: NRPhyConfig, output_path: str) -> None:
    """Visualize DL/UL slots and symbol-dependent CP construction."""
    _ensure_matplotlib()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    total_slots = cfg.n_dl_slots + cfg.n_ul_slots
    fig, ax = plt.subplots(figsize=(13, 3.8), constrained_layout=True)
    x = 0.0
    symbol_id = 0
    for slot_idx in range(total_slots):
        link = "DL" if slot_idx < cfg.n_dl_slots else "UL"
        color = "#2563eb" if link == "DL" else "#16a34a"
        for local_symbol, cp_len in enumerate(cfg.cp_lengths):
            ax.broken_barh([(x, cp_len)], (0.2, 0.18), facecolors="#9ca3af")
            ax.broken_barh([(x + cp_len, cfg.n_fft)], (0.45, 0.35), facecolors=color, alpha=0.85)
            if local_symbol in ResourceGridMapper.pilot_symbols:
                ax.broken_barh([(x + cp_len, cfg.n_fft)], (0.85, 0.12), facecolors="#ef4444")
            x += cp_len + cfg.n_fft
            symbol_id += 1
        ax.axvline(x, color="black", linewidth=0.8, alpha=0.5)
        ax.text(
            x - cfg.slot_samples / 2,
            1.08,
            f"{link} slot {slot_idx if link == 'DL' else slot_idx - cfg.n_dl_slots}",
            ha="center",
            va="bottom",
        )

    ax.set_title("TDD frame construction: CP + useful OFDM symbols, pilots marked in red")
    ax.set_xlabel("Sample index")
    ax.set_yticks([0.29, 0.62, 0.91])
    ax.set_yticklabels(["CP", "OFDM", "Pilot"])
    ax.set_xlim(0, x)
    ax.set_ylim(0.1, 1.25)
    ax.grid(axis="x", alpha=0.25)
    ax.text(
        0.0,
        -0.18,
        f"Total OFDM symbols: {symbol_id}, slot samples: {cfg.slot_samples}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        clip_on=False,
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
