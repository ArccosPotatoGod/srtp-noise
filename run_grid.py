#!/usr/bin/env python3
"""Batch grid experiment entry point."""

import argparse

from runner import ExperimentRunner
from src.report import format_summary


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

    runner = ExperimentRunner(args.config, args.grid)
    runner.run(n_runs=args.runs)

    print(f"\nCompleted {len(runner.results)} experiments.\n")
    print(format_summary(runner.results))

    runner.export_results(args.output)
    print(f"Results exported to {args.output}")

    runner.export_text_report("results/report.txt")
    print("Text report saved to results/report.txt")

    # Raw metric plots (strategy only — no denoiser grouping)
    runner.plot_snr_vs_distance("results/snr_vs_distance.png")
    runner.plot_cwer_vs_distance("results/cwer_vs_distance.png")

    # Enhanced metric plots (per denoiser, with cancel/no-cancel distinction) for each strategy
    strategies = ["off", "fixed_weight", "gaussian_baseline", "sweeping_baseline", "hopping_baseline"]
    for strat in strategies:
        runner.plot_snr_enhanced_vs_distance(
            f"results/snr_enhanced_{strat}.png", strategy=strat)
        runner.plot_cwer_enhanced_vs_distance(
            f"results/cwer_enhanced_{strat}.png", strategy=strat)

    # Denoiser comparison bar charts at key distances (with cancel/no-cancel distinction)
    for dist in [1.0, 3.0]:
        runner.plot_denoiser_comparison(
            f"results/denoiser_comparison_{dist:.0f}m.png", distance=dist)

    # Scenario comparison: noise+cancel vs noise only (no cancel) vs no jammer
    runner.plot_scenario_comparison("results/scenario_comparison.png")

    # CWER scatter: raw vs enhanced
    runner.plot_cwer_scatter("results/cwer_scatter.png")

    print("Plots saved to results/")

    # Optional: heatmap
    if args.heatmap:
        runner.run_coverage_heatmap(
            resolution=args.heatmap_res,
            strategy="fixed_weight",
            denoiser="none",
            path="results/coverage_heatmap.png",
        )
        print("Coverage heatmap saved to results/coverage_heatmap.png")

    # Optional: angle sweep
    if args.angle_sweep:
        runner.run_angle_sweep(
            path="results/angle_sweep.png",
            distance=2.0,
            strategy="fixed_weight",
            denoiser="none",
        )
        print("Angle sweep saved to results/angle_sweep.png")

    # Optional: pipeline audio visualization
    if args.export_audio:
        runner.save_pipeline_audio_plots(output_dir="results/audio_stages")
        print("Pipeline audio plots saved to results/audio_stages/")


if __name__ == "__main__":
    main()
