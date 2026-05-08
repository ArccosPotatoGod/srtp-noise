#!/usr/bin/env python3
"""Batch grid experiment entry point."""

import argparse
import hashlib
import itertools
from typing import Any, Dict, List

import numpy as np
from tqdm import tqdm

from src.config import load_config, load_grid_config, apply_overrides
from runner import run_single
from src.exporter import (
    export_csv,
    export_text_report,
    plot_snr_vs_distance,
    plot_cwer_vs_distance,
    plot_snr_enhanced_vs_distance,
    plot_cwer_enhanced_vs_distance,
    plot_denoiser_comparison,
    plot_cwer_scatter,
    plot_scenario_comparison,
)
from src.report import format_summary


def _expand_grid(grid: Dict) -> List[Dict[str, Any]]:
    """Cartesian product of grid parameter lists."""
    if not grid:
        return [{}]
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _seed_from_override(override: Dict) -> int:
    """Deterministic seed from signal-generation params only (excludes attacker.*)."""
    signal_keys = {k: v for k, v in override.items()
                   if not k.startswith("attacker.")}
    seed_str = str(sorted(signal_keys.items()))
    return int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16) % (2 ** 31)


def main():
    parser = argparse.ArgumentParser(description="MicFrozen batch grid experiment")
    parser.add_argument("--config", default="configs/base.yaml",
                        help="Base YAML config file")
    parser.add_argument("--grid", default="configs/grid.yaml",
                        help="Grid YAML config file")
    parser.add_argument("--output", default="results/exp.csv",
                        help="Output CSV path for results")
    parser.add_argument("--heatmap", action="store_true",
                        help="Generate 2D coverage heatmap (slower)")
    parser.add_argument("--heatmap-res", type=int, default=10,
                        help="Coverage heatmap resolution (default: 10)")
    parser.add_argument("--angle-sweep", action="store_true",
                        help="Generate angle sweep plot")
    parser.add_argument("--export-audio", action="store_true",
                        help="Export pipeline waveform & spectrogram plots")
    parser.add_argument("--runs", type=int, default=1,
                        help="Number of ASR corruption seeds per config (default: 1)")
    args = parser.parse_args()

    base_config = load_config(args.config)
    grid = load_grid_config(args.grid)

    # Expand and run
    combinations = _expand_grid(grid)
    results = []

    for override in tqdm(combinations, desc="Running experiments"):
        config = apply_overrides(base_config, override)
        seed = _seed_from_override(override)

        if args.runs <= 1:
            metrics = run_single(config, seed)
            metrics.update(override)
            results.append(metrics)
        else:
            all_metrics = []
            for run_idx in range(args.runs):
                m = run_single(config, seed, run_index=run_idx)
                all_metrics.append(m)
            avg = {k: float(np.mean([m[k] for m in all_metrics]))
                   for k in all_metrics[0]}
            avg.update(override)
            results.append(avg)

    print(f"\nCompleted {len(results)} experiments.\n")
    print(format_summary(results))

    # --- CSV & text report ---
    export_csv(results, args.output)
    print(f"Results exported to {args.output}")

    export_text_report(results, "results/report.txt")
    print("Text report saved to results/report.txt")

    # --- Plots ---
    plot_snr_vs_distance(results, "results/snr_vs_distance.png")
    plot_cwer_vs_distance(results, "results/cwer_vs_distance.png")

    strategies = ["off", "fixed_weight", "gaussian_baseline",
                  "sweeping_baseline", "hopping_baseline"]
    for strat in strategies:
        plot_snr_enhanced_vs_distance(
            results, f"results/snr_enhanced_{strat}.png", strategy=strat)
        plot_cwer_enhanced_vs_distance(
            results, f"results/cwer_enhanced_{strat}.png", strategy=strat)

    for dist in [1.0, 3.0]:
        plot_denoiser_comparison(
            results, f"results/denoiser_comparison_{dist:.0f}m.png", distance=dist)

    plot_scenario_comparison(results, "results/scenario_comparison.png")
    plot_cwer_scatter(results, "results/cwer_scatter.png")

    print("Plots saved to results/")

    # --- Optional: heatmap ---
    if args.heatmap:
        from src.analysis import run_coverage_heatmap
        run_coverage_heatmap(
            run_single, base_config,
            path="results/coverage_heatmap.png",
            resolution=args.heatmap_res,
            strategy="fixed_weight",
            denoiser="none",
        )
        print("Coverage heatmap saved to results/coverage_heatmap.png")

    # --- Optional: angle sweep ---
    if args.angle_sweep:
        from src.analysis import run_angle_sweep
        run_angle_sweep(
            run_single, base_config,
            path="results/angle_sweep.png",
            distance=2.0,
            strategy="fixed_weight",
            denoiser="none",
        )
        print("Angle sweep saved to results/angle_sweep.png")

    # --- Optional: pipeline audio plots ---
    if args.export_audio:
        from scipy.signal import fftconvolve
        from src.channel import ChannelModule
        from src.exporter import save_pipeline_audio_plots
        from strategies.nonlinearity import create_nonlinearity

        config = apply_overrides(base_config, {})
        sig = run_single(config, seed=42, return_signals=True)

        channel = ChannelModule(config)
        audible_rirs, ultrasonic_rirs = channel.compute_rir()
        ref_sig = fftconvolve(sig["s_src"], audible_rirs["src_to_ref"])[:len(sig["s_src"])]

        nonlinearity = create_nonlinearity(config.spy_mic.nonlinearity,
                                           config.spy_mic.nonlinearity_params)
        audible_arrival = fftconvolve(sig["s_src"],
                                      audible_rirs["src_to_spy"][0])[:len(sig["s_src"])]
        jammer_baseband = sig["s_cancel"] + sig["n_coherent"]
        ultrasonic_arrival = fftconvolve(jammer_baseband,
                                         ultrasonic_rirs["jammer_to_spy"][0])[:len(sig["s_src"])]
        ultrasonic_demod = nonlinearity.apply(ultrasonic_arrival)

        plot_signals = {
            "s_src": sig["s_src"],
            "ref_sig": ref_sig,
            "s_cancel": sig["s_cancel"],
            "n_coherent": sig["n_coherent"],
            "audible_arrival": audible_arrival,
            "ultrasonic_arrival": ultrasonic_arrival,
            "ultrasonic_demod": ultrasonic_demod,
            "spy_rec": sig["spy_rec"],
            "enhanced": sig["enhanced"],
            "jammer_baseband": jammer_baseband,
            "noise_ref": sig["noise_ref"],
        }
        save_pipeline_audio_plots(plot_signals, "results/audio_stages",
                                  config.sim.fs, denoiser_name=config.attacker.denoiser)
        print("Pipeline audio plots saved to results/audio_stages/")


if __name__ == "__main__":
    main()
