#!/usr/bin/env python3
"""MicFrozen Simulation — Main Experiment Pipeline.

Runs the full experiment:
1. Load/prepare audio data
2. Apply MicFrozen cancellation + noise jamming
3. Apply adversary denoising attacks (filter, ICA, beamforming)
4. Compute metrics (SNR, MFCC distance)
5. Generate result charts

Usage:
    python demo_main.py                    # Run full pipeline
    python demo_main.py --single           # Single-sample demo
    python demo_main.py --output results/  # Specify output dir
"""

import os
import sys
import argparse
import numpy as np
import soundfile as sf
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.propagation import propagate_signal
from src.demodulation import nonlinear_demod
from src.cancellation import apply_cancellation
from src.noise import generate_coherent_noise, adaptive_coherent_noise, generate_gaussian_noise
from src.attacks import apply_bandstop, ica_denoise, select_speech_component, delay_and_sum
from src.metrics import compute_snr, compute_mfcc_distance
from src.visualize import plot_waveforms, plot_snr_vs_distance


# Default parameters
FS = 16000
SRC_POS = (0.0, 0.0)
JAMMER_POS = (0.2, 0.0)

DISTANCES = [1.0, 2.0, 3.0, 4.0, 5.0]
ANGLES = [0, 15, 30, 45]
METHODS = ["gaussian", "coherent_fixed", "adaptive"]
ATTACKS = ["none", "bandstop", "ica", "beamforming"]


def load_or_generate_test_signal(data_dir="data"):
    """Load a real audio sample or generate a synthetic one."""
    list_path = os.path.join(data_dir, "file_list.txt")
    if os.path.exists(list_path):
        with open(list_path) as f:
            paths = [line.strip() for line in f if line.strip()]
        if paths:
            s, fs = sf.read(paths[0])
            if fs != FS:
                import librosa
                s = librosa.resample(s, orig_sr=fs, target_sr=FS)
            return s.astype(np.float64), fs if fs == FS else FS

    # Fallback: synthetic
    from scripts.download_data import generate_synthetic_signals
    paths = generate_synthetic_signals(data_dir, 1)
    s, _ = sf.read(paths[0])
    return s.astype(np.float64), FS


def run_single_demo(s, spy_distance=2.0, angle_deg=15.0, output_dir="results"):
    """Run a single-sample demonstration with plots."""
    os.makedirs(output_dir, exist_ok=True)

    angle_rad = np.deg2rad(angle_deg)
    spy_pos = (spy_distance * np.cos(angle_rad),
               spy_distance * np.sin(angle_rad))

    # 1. Apply cancellation
    residual, s_direct = apply_cancellation(s, SRC_POS, JAMMER_POS, spy_pos, FS)

    # 2. Generate noises
    gaussian_noise = generate_gaussian_noise(len(s), FS)
    coherent_noise = generate_coherent_noise(s, FS)
    adaptive_noise = adaptive_coherent_noise(s, spy_distance, FS)

    # 3. Mix with residuals
    mixed_gauss = residual + gaussian_noise[:len(residual)]
    mixed_coherent = residual + coherent_noise[:len(residual)]
    mixed_adaptive = residual + adaptive_noise[:len(residual)]

    # 4. Plot waveforms
    signals = [s, s_direct, residual, mixed_gauss, mixed_coherent, mixed_adaptive]
    labels = ["Original s(t)", "Direct at spy", "After cancellation",
              "+ Gaussian noise", "+ Coherent noise", "+ Adaptive noise"]

    print(f"\n=== Single Sample Demo (dist={spy_distance}m, angle={angle_deg}°) ===")
    print(f"{'Signal':<30} {'SNR (dB)':>10}")
    print("-" * 42)
    for name, sig in zip(labels[1:], signals[1:]):
        snr_val = compute_snr(s, sig)
        print(f"{name:<30} {snr_val:>10.2f}")

    fig = plot_waveforms(signals, labels, FS)
    fig.savefig(os.path.join(output_dir, "waveform_demo.png"), dpi=150)
    print(f"\nWaveform plot saved to {output_dir}/waveform_demo.png")


def run_batch_experiments(s, output_dir="results"):
    """Run full experiment matrix and collect metrics."""
    os.makedirs(output_dir, exist_ok=True)
    results = []

    print(f"\n=== Batch Experiment ===")
    print(f"Distances: {DISTANCES}")
    print(f"Angles: {ANGLES}")
    print(f"Methods: {METHODS}")
    print(f"Attacks: {ATTACKS}")
    print("-" * 60)

    for distance in DISTANCES:
        for angle_deg in ANGLES:
            angle_rad = np.deg2rad(angle_deg)
            spy_pos = (distance * np.cos(angle_rad),
                       distance * np.sin(angle_rad))

            # Apply cancellation
            residual, s_direct = apply_cancellation(s, SRC_POS, JAMMER_POS, spy_pos, FS)

            # Generate noise types
            noises = {
                "gaussian": generate_gaussian_noise(len(s), FS),
                "coherent_fixed": generate_coherent_noise(s, FS),
                "adaptive": adaptive_coherent_noise(s, distance, FS),
            }

            for method_name, noise in noises.items():
                mixed = residual + noise[:len(residual)]

                for attack_name in ATTACKS:
                    # Apply attack
                    if attack_name == "none":
                        processed = mixed
                    elif attack_name == "bandstop":
                        processed = apply_bandstop(mixed, FS)
                    elif attack_name == "ica":
                        # Generate second channel (different position)
                        spy_pos2 = (distance * np.cos(angle_rad + 0.1),
                                    distance * np.sin(angle_rad + 0.1))
                        residual2, _ = apply_cancellation(s, SRC_POS, JAMMER_POS, spy_pos2, FS)
                        mixed2 = residual2 + noise[:len(residual2)]
                        S_ = ica_denoise(mixed, mixed2)
                        processed = select_speech_component(S_, s)
                    elif attack_name == "beamforming":
                        # Simulate 4-mic linear array
                        delays = [0, -2, -4, -6]
                        channels = np.array([np.roll(mixed, d) for d in delays])
                        processed = delay_and_sum(channels, [0, 0, 0, 0])

                    snr_val = compute_snr(s, processed)
                    mfcc_dist = compute_mfcc_distance(s, processed, FS)

                    results.append({
                        "distance": distance,
                        "angle": angle_deg,
                        "method": method_name,
                        "attack": attack_name,
                        "snr": snr_val,
                        "mfcc_distance": mfcc_dist,
                    })

                    print(f"d={distance:.0f}m a={angle_deg:>2}deg "
                          f"method={method_name:<15} attack={attack_name:<12} "
                          f"SNR={snr_val:.2f}dB")

    # Save results
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # Generate summary plots
    fig = plot_snr_vs_distance(
        df[df["attack"] == "none"],
        os.path.join(output_dir, "snr_vs_distance.png")
    )
    print(f"SNR plot saved to {output_dir}/snr_vs_distance.png")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="MicFrozen Simulation — Experiment Pipeline"
    )
    parser.add_argument("--single", action="store_true",
                        help="Run single-sample demo only")
    parser.add_argument("--distance", type=float, default=2.0,
                        help="Eavesdropper distance for single demo (m)")
    parser.add_argument("--angle", type=float, default=15.0,
                        help="Angle offset for single demo (deg)")
    parser.add_argument("--output", type=str, default="results",
                        help="Output directory for results")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Directory for audio data")
    args = parser.parse_args()

    print("=" * 60)
    print("MicFrozen Simulation — Speech Privacy Protection")
    print("Based on Gao et al., MobiCom 2023")
    print("=" * 60)

    # Load data
    print("\n[1/3] Loading test signal...")
    s, fs = load_or_generate_test_signal(args.data_dir)
    print(f"      Signal: {len(s)/fs:.2f}s @ {fs}Hz")

    if args.single:
        print("\n[2/3] Running single-sample demo...")
        run_single_demo(s, args.distance, args.angle, args.output)
    else:
        print("\n[2/3] Running batch experiments...")
        run_batch_experiments(s, args.output)

    print("\n[3/3] Done.")
    print(f"Results in: {os.path.abspath(args.output)}/")


if __name__ == "__main__":
    main()
