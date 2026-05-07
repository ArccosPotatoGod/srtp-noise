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
    args = parser.parse_args()

    runner = ExperimentRunner(args.config, args.grid)
    runner.run()

    print(f"\nCompleted {len(runner.results)} experiments.\n")
    print(format_summary(runner.results))

    runner.export_results(args.output)
    print(f"Results exported to {args.output}")

    runner.plot_snr_vs_distance("results/snr_vs_distance.png")
    runner.plot_cwer_vs_distance("results/cwer_vs_distance.png")
    print("Plots saved to results/")


if __name__ == "__main__":
    main()
