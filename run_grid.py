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
    args = parser.parse_args()

    runner = ExperimentRunner(args.config, args.grid)
    runner.run()

    print(f"\nCompleted {len(runner.results)} experiments.\n")
    print(format_summary(runner.results))

    runner.export_results(args.output)
    print(f"Results exported to {args.output}")

    runner.export_text_report("results/report.txt")
    print("Text report saved to results/report.txt")

    runner.plot_snr_vs_distance("results/snr_vs_distance.png")
    runner.plot_cwer_vs_distance("results/cwer_vs_distance.png")
    print("Plots saved to results/")

    if args.heatmap:
        runner.run_coverage_heatmap(
            resolution=args.heatmap_res,
            strategy="fixed_weight",
            denoiser="none",
            path="results/coverage_heatmap.png",
        )
        print("Coverage heatmap saved to results/coverage_heatmap.png")


if __name__ == "__main__":
    main()
