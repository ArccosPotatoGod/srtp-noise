"""Analysis — spatial sweeps (heatmap, angle sweep) that call run_single repeatedly."""

import copy
from pathlib import Path
from typing import Callable, Dict, List

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src.config import ScenarioConfig


def run_coverage_heatmap(run_fn: Callable[[ScenarioConfig, int, int], Dict],
                         base_config: ScenarioConfig,
                         path: str = "results/coverage_heatmap.png",
                         resolution: int = 10,
                         z: float = 1.5,
                         strategy: str = "fixed_weight",
                         denoiser: str = "none") -> None:
    """2D spatial sweep: scan spy mic across the room x-y plane.

    Corresponds to the coverage heatmap in paper Fig.11.
    """
    room = base_config.room
    src = base_config.source.pos
    xs = np.linspace(0.5, room.dim[0] - 0.5, resolution)
    ys = np.linspace(0.5, room.dim[1] - 0.5, resolution)
    snr_grid = np.zeros((resolution, resolution))

    config = copy.deepcopy(base_config)
    config.jammer.coherent_strategy = strategy
    config.attacker.denoiser = denoiser

    total = resolution * resolution
    for i, spy_x in enumerate(tqdm(xs, desc="Heatmap X")):
        for j, spy_y in enumerate(ys):
            config.spy_mic.positions = [
                (float(spy_x), float(spy_y), z),
                (float(spy_x), float(spy_y) + 0.15, z),
            ]
            seed = hash((spy_x, spy_y, strategy, denoiser)) & 0x7FFFFFFF
            metrics = run_fn(config, seed)
            snr_grid[j, i] = metrics["snr_raw"]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.pcolormesh(xs, ys, snr_grid, shading="auto", cmap="RdYlGn")
    ax.scatter(src[0], src[1], marker="*", s=200, color="black",
               label="Source")
    ax.scatter(base_config.jammer.pos_spk[0],
               base_config.jammer.pos_spk[1],
               marker="s", s=100, color="blue", label="Jammer")
    cbar = fig.colorbar(im, ax=ax, label="SNR (dB)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"Coverage Heatmap — {strategy} | {denoiser}\n"
                 f"SNR at z={z} m, res={resolution}x{resolution}")
    ax.legend(fontsize=7)
    ax.set_aspect("equal")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_angle_sweep(run_fn: Callable[[ScenarioConfig, int, int], Dict],
                    base_config: ScenarioConfig,
                    path: str = "results/angle_sweep.png",
                    distance: float = 2.0,
                    angles: List[float] = None,
                    strategy: str = "fixed_weight",
                    denoiser: str = "none") -> None:
    """CWER vs angle sweep at a fixed distance (paper Fig.11 angle dimension)."""
    if angles is None:
        angles = [0, 10, 20, 30, 40, 50, 60]

    config = copy.deepcopy(base_config)
    config.jammer.coherent_strategy = strategy
    config.attacker.denoiser = denoiser

    cwer_raw_vals = []
    cwer_enh_vals = []
    snr_raw_vals = []

    for angle in tqdm(angles, desc="Angle sweep"):
        src = base_config.source.pos
        base_pos = base_config.spy_mic.positions[0]
        rad = np.radians(angle)
        new_x = src[0] + distance * np.cos(rad)
        new_y = src[1] + distance * np.sin(rad)
        config.spy_mic.positions = [
            (new_x, new_y, base_pos[2]),
            (new_x, new_y + 0.15, base_pos[2]),
        ]
        seed = hash((distance, angle, strategy, denoiser)) & 0x7FFFFFFF
        metrics = run_fn(config, seed)
        cwer_raw_vals.append(metrics["cwer_raw"])
        cwer_enh_vals.append(metrics["cwer_enhanced"])
        snr_raw_vals.append(metrics["snr_raw"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(angles, cwer_raw_vals, "o-", label="CWER raw", color="tab:red")
    ax1.plot(angles, cwer_enh_vals, "s-", label="CWER enhanced", color="tab:blue")
    ax1.set_xlabel("Angle (degrees)")
    ax1.set_ylabel("CWER (%)")
    ax1.set_title(f"Angle Sweep — {strategy} | {denoiser} @ {distance:.0f} m")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(angles, snr_raw_vals, "D-", color="tab:orange")
    ax2.set_xlabel("Angle (degrees)")
    ax2.set_ylabel("SNR (dB)")
    ax2.set_title(f"SNR vs Angle @ {distance:.0f} m")
    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
