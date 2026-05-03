#!/usr/bin/env python3
"""MicFrozen Simulation — Main Experiment Pipeline.

Implements the full experiment workflow from Gao et al., MobiCom 2023:
1. Load/prepare audio data (LibriSpeech test-clean or synthetic)
2. Apply MicFrozen cancellation + noise jamming
3. Apply adversary denoising attacks (filter, ICA, beamforming)
4. Compute metrics (SNR, MFCC distance, WER)
5. Generate result charts per spec Sections 7.1-7.4

Usage:
    python demo_main.py                     # Full batch experiment
    python demo_main.py --single            # Single-sample demo with plots
    python demo_main.py --single --load data/sample.wav  # Use real audio
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.propagation import propagate_signal
from src.cancellation import apply_cancellation
from src.noise import (
    generate_coherent_noise,
    adaptive_coherent_noise,
    generate_gaussian_noise,
)
from src.attacks import (
    apply_bandstop,
    apply_bandpass,
    ica_denoise,
    select_speech_component,
    delay_and_sum,
)
from src.metrics import compute_snr, compute_mfcc_distance, compute_segment_snr
from src.visualize import (
    plot_waveforms,
    plot_spectrogram_comparison,
    plot_snr_vs_distance,
    plot_wer_bars,
    plot_heatmap_grid,
)

# ---------------------------------------------------------------------------
# Default parameters (from spec Section 4.1)
# ---------------------------------------------------------------------------
FS = 16000
SPEED_OF_SOUND = 340.0
SRC_POS = (0.0, 0.0)        # Speaker position
JAMMER_POS = (0.2, 0.0)     # MicFrozen device position
REF_MIC_POS = (0.1, 0.0)    # Reference microphone (close to source)

# Experiment matrix (spec Section 7.3)
DISTANCES = [1.0, 2.0, 3.0, 4.0, 5.0]
ANGLES = [0, 15, 30, 45]
METHODS = ["gaussian", "coherent_fixed", "adaptive"]
ATTACKS = ["none", "bandstop", "bandpass", "ica", "beamforming"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_or_generate_test_signal(data_dir="data", file_index=0):
    """Load a real audio sample or fall back to synthetic."""
    list_path = os.path.join(data_dir, "file_list.txt")
    if os.path.exists(list_path):
        with open(list_path) as f:
            paths = [line.strip() for line in f if line.strip()]
        if paths and file_index < len(paths):
            s, fs_in = sf.read(paths[file_index])
            if fs_in != FS:
                import librosa
                s = librosa.resample(s, orig_sr=fs_in, target_sr=FS)
            return s.astype(np.float32), FS

    # Fallback: synthetic test signal
    from scripts.download_data import generate_synthetic_signals
    paths = generate_synthetic_signals(data_dir, max(file_index + 1, 5))
    s, _ = sf.read(paths[file_index])
    return s.astype(np.float32), FS


def load_specific_file(filepath):
    """Load a specific audio file and resample if needed."""
    s, fs_in = sf.read(filepath)
    if fs_in != FS:
        import librosa
        s = librosa.resample(s, orig_sr=fs_in, target_sr=FS)
    return s.astype(np.float32), FS


# ---------------------------------------------------------------------------
# Core experiment functions
# ---------------------------------------------------------------------------

def spy_position(distance, angle_deg):
    """Convert polar (distance, angle) to Cartesian position."""
    rad = np.deg2rad(angle_deg)
    return (distance * np.cos(rad), distance * np.sin(rad))


def generate_all_noises(s, distance, rng=None):
    """Generate all three noise types for a given distance."""
    T = len(s)
    return {
        "gaussian": generate_gaussian_noise(T, FS, rng=rng),
        "coherent_fixed": generate_coherent_noise(s, FS, rng=rng),
        "adaptive": adaptive_coherent_noise(s, distance, FS, rng=rng),
    }


def apply_attack(mixed, s_original, attack_name, distance, angle_rad):
    """Apply a single denoising attack to the mixed signal."""
    if attack_name == "none":
        return mixed

    elif attack_name == "bandstop":
        return apply_bandstop(mixed, FS)

    elif attack_name == "bandpass":
        return apply_bandpass(mixed, FS)

    elif attack_name == "ica":
        # Second channel: slightly different spy position
        spy2 = (distance * np.cos(angle_rad + 0.1),
                distance * np.sin(angle_rad + 0.1))
        res2, _ = apply_cancellation(s_original, SRC_POS, JAMMER_POS, spy2, FS)
        # Regenerate same-type noise for second channel
        rng2 = np.random.default_rng(42)
        n2 = generate_coherent_noise(s_original, FS, rng=rng2)
        mixed2 = res2 + n2[:len(res2)]
        S_ = ica_denoise(mixed, mixed2)
        return select_speech_component(S_, s_original)

    elif attack_name == "beamforming":
        # 4-mic linear array, 0.05m spacing (spec Section 5.3)
        mic_spacing = 0.05
        delays_samples = []
        for i in range(4):
            pos = (distance + i * mic_spacing,
                   distance * np.tan(angle_rad) if angle_rad != 0 else 0)
            d = np.linalg.norm(np.array(pos) - np.array(SRC_POS))
            delays_samples.append(-int(round(d / SPEED_OF_SOUND * FS)))
        # Normalize delays relative to first mic
        delays_samples = [d - delays_samples[0] for d in delays_samples]
        channels = np.zeros((4, len(mixed)))
        for i, d in enumerate(delays_samples):
            channels[i] = np.roll(mixed, d)
        return delay_and_sum(channels, [0, 0, 0, 0])

    else:
        raise ValueError(f"Unknown attack: {attack_name}")


# ---------------------------------------------------------------------------
# Single-sample demo (spec Section 7.1)
# ---------------------------------------------------------------------------

def run_single_demo(s, spy_distance=2.0, angle_deg=15.0, output_dir="results"):
    """Run single-sample demo with waveform + spectrogram plots."""
    os.makedirs(output_dir, exist_ok=True)
    spy_pos = spy_position(spy_distance, angle_deg)

    # 1. Cancellation
    residual, s_direct = apply_cancellation(s, SRC_POS, JAMMER_POS, spy_pos, FS)

    # 2. Generate noises
    rng = np.random.default_rng(42)
    noises = generate_all_noises(s, spy_distance, rng=rng)

    # 3. Mix
    mixed = {}
    for name in METHODS:
        n = noises[name][:len(residual)]
        mixed[name] = residual + n

    # 4. Print SNR table
    print(f"\n{'='*60}")
    print(f"  Single Sample Demo  |  distance = {spy_distance}m  angle = {angle_deg}°")
    print(f"{'='*60}")
    print(f"{'Signal':<35} {'SNR (dB)':>10} {'SegSNR (dB)':>12}")
    print(f"{'-'*57}")

    all_signals = {
        "Original s(t)": s,
        "Direct at spy": s_direct,
        "After cancellation": residual,
    }
    for name, sig in all_signals.items():
        snr = compute_snr(s, sig)
        seg = compute_segment_snr(s, sig)
        print(f"{name:<35} {snr:>10.2f} {seg:>12.2f}")

    for method_name in METHODS:
        label = f"+ {method_name}"
        snr = compute_snr(s, mixed[method_name])
        seg = compute_segment_snr(s, mixed[method_name])
        print(f"{label:<35} {snr:>10.2f} {seg:>12.2f}")

    # Also show after-attack results
    print(f"\n{'After denoising attacks:':<35} {'SNR (dB)':>10} {'SegSNR (dB)':>12}")
    print(f"{'-'*57}")
    coherent_mixed = mixed["coherent_fixed"]
    for att in ["bandstop", "bandpass", "ica", "beamforming"]:
        angle_rad = np.deg2rad(angle_deg)
        processed = apply_attack(coherent_mixed, s, att, spy_distance, angle_rad)
        snr = compute_snr(s, processed)
        seg = compute_segment_snr(s, processed)
        print(f"  {att:<33} {snr:>10.2f} {seg:>12.2f}")

    # 5. Waveform plot
    wave_signals = [s, s_direct, residual,
                    mixed["gaussian"], mixed["coherent_fixed"], mixed["adaptive"]]
    wave_labels = ["Original s(t)", "Direct at spy", "After cancellation",
                   "+ Gaussian noise (UMJ)", "+ Coherent noise (MicFrozen)",
                   "+ Adaptive noise (Ours)"]
    fig_wave = plot_waveforms(wave_signals, wave_labels, FS,
                              title=f"Waveforms (d={spy_distance}m, {angle_deg}°)")
    wave_path = os.path.join(output_dir, "waveform_demo.png")
    fig_wave.savefig(wave_path, dpi=150, bbox_inches="tight")
    plt.close(fig_wave)
    print(f"\nWaveform plot -> {wave_path}")

    # 6. Spectrogram comparison
    spec_signals = [s, mixed["gaussian"], mixed["coherent_fixed"], mixed["adaptive"]]
    spec_labels = ["Original Speech", "+ Gaussian (UMJ)",
                   "+ Coherent (MicFrozen)", "+ Adaptive (Ours)"]

    # Also add post-ICA
    angle_rad = np.deg2rad(angle_deg)
    ica_result = apply_attack(coherent_mixed, s, "ica", spy_distance, angle_rad)
    spec_signals.append(ica_result)
    spec_labels.append("Coherent → ICA denoised")

    fig_spec = plot_spectrogram_comparison(
        spec_signals, spec_labels, FS,
        title=f"Spectrograms (d={spy_distance}m, {angle_deg}°)",
        output_path=os.path.join(output_dir, "spectrogram_demo.png"))
    plt.close(fig_spec)
    print(f"Spectrogram plot -> {output_dir}/spectrogram_demo.png")

    return mixed, noises


# ---------------------------------------------------------------------------
# Batch experiments (spec Section 7.3)
# ---------------------------------------------------------------------------

def run_batch_experiments(s, output_dir="results"):
    """Run full experiment matrix: distances × angles × methods × attacks."""
    os.makedirs(output_dir, exist_ok=True)
    results = []

    total = len(DISTANCES) * len(ANGLES) * len(METHODS) * len(ATTACKS)
    print(f"\n{'='*60}")
    print(f"  Batch Experiment  |  {total} combinations")
    print(f"  Distances: {DISTANCES}")
    print(f"  Angles: {ANGLES}")
    print(f"  Methods: {METHODS}")
    print(f"  Attacks: {ATTACKS}")
    print(f"{'='*60}")

    count = 0
    for distance in DISTANCES:
        for angle_deg in ANGLES:
            angle_rad = np.deg2rad(angle_deg)
            spy_pos = spy_position(distance, angle_deg)

            residual, s_direct = apply_cancellation(
                s, SRC_POS, JAMMER_POS, spy_pos, FS)

            rng = np.random.default_rng(42)
            noises = generate_all_noises(s, distance, rng=rng)

            for method_name in METHODS:
                noise = noises[method_name]
                mixed = residual + noise[:len(residual)]

                for attack_name in ATTACKS:
                    processed = apply_attack(
                        mixed, s, attack_name, distance, angle_rad)

                    snr_val = compute_snr(s, processed)
                    mfcc_dist = compute_mfcc_distance(s, processed, FS)
                    seg_snr = compute_segment_snr(s, processed)

                    results.append({
                        "distance": distance,
                        "angle": angle_deg,
                        "method": method_name,
                        "attack": attack_name,
                        "snr": snr_val,
                        "seg_snr": seg_snr,
                        "mfcc_distance": mfcc_dist,
                    })

                    count += 1
                    if count % 40 == 0 or count == total:
                        print(f"  [{count}/{total}] d={distance:.0f}m "
                              f"a={angle_deg:>2}° {method_name:<15} "
                              f"{attack_name:<12} SNR={snr_val:.2f}")

    # Save CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults -> {csv_path}  ({len(df)} rows)")

    return df


# ---------------------------------------------------------------------------
# Result visualization (spec Section 7.4)
# ---------------------------------------------------------------------------

def generate_summary_charts(df, output_dir="results"):
    """Generate all summary charts from experiment results."""
    os.makedirs(output_dir, exist_ok=True)

    print("\n--- Generating Charts ---")

    # 1. SNR vs Distance (no attack)
    df_no_attack = df[df["attack"] == "none"]
    fig = plot_snr_vs_distance(
        df_no_attack,
        os.path.join(output_dir, "snr_vs_distance.png"))
    plt.close(fig)
    print("  [1/5] SNR vs Distance -> snr_vs_distance.png")

    # 2. SNR vs Distance after filtering
    for att in ["bandstop", "bandpass", "ica", "beamforming"]:
        df_att = df[df["attack"] == att]
        if len(df_att) == 0:
            continue
        fig = plot_snr_vs_distance(
            df_att,
            os.path.join(output_dir, f"snr_vs_distance_{att}.png"))
        plt.close(fig)
    print("  [2/5] SNR vs Distance (per attack) done")

    # 3. MFCC distance bar chart
    df_avg = df.groupby(["method", "attack"], as_index=False)["mfcc_distance"].mean()
    fig = plot_wer_bars(df_avg, os.path.join(output_dir, "mfcc_bars.png"))
    plt.close(fig)
    print("  [3/5] MFCC distance bars -> mfcc_bars.png")

    # 4. SNR heatmaps (distance × angle) for each method
    matrices = {}
    for method in METHODS:
        sub = df[(df["method"] == method) & (df["attack"] == "none")]
        pivot = sub.pivot_table(values="snr", index="distance",
                                columns="angle", aggfunc="mean")
        matrices[method] = pivot.values

    fig = plot_heatmap_grid(
        matrices, METHODS, DISTANCES, ANGLES,
        os.path.join(output_dir, "snr_heatmaps.png"))
    plt.close(fig)
    print("  [4/5] SNR heatmaps -> snr_heatmaps.png")

    # 5. Summary text report
    report_path = os.path.join(output_dir, "summary.txt")
    write_summary_report(df, report_path)
    print(f"  [5/5] Summary report -> summary.txt")

    print("\nAll charts generated in:", os.path.abspath(output_dir))


def write_summary_report(df, path):
    """Write a human-readable summary of key findings."""
    lines = []
    lines.append("=" * 60)
    lines.append("MicFrozen Simulation — Experiment Summary")
    lines.append("Based on Gao et al., MobiCom 2023")
    lines.append("=" * 60)
    lines.append("")

    # Best jamming method (lowest SNR = best privacy)
    lines.append("--- Jamming Effectiveness (SNR, lower = better) ---")
    for method in METHODS:
        sub = df[(df["method"] == method) & (df["attack"] == "none")]
        avg_snr = sub["snr"].mean()
        lines.append(f"  {method:<20} avg SNR = {avg_snr:.2f} dB")

    lines.append("")
    lines.append("--- Attack Resilience (SNR after denoising) ---")
    for att in ATTACKS:
        if att == "none":
            continue
        sub = df[df["attack"] == att]
        if len(sub) == 0:
            continue
        avg_snr = sub["snr"].mean()
        lines.append(f"  After {att:<15} avg SNR = {avg_snr:.2f} dB")

    lines.append("")
    lines.append("--- Adaptive vs Fixed Coherent (SNR at far range >= 4m) ---")
    far = df[(df["distance"] >= 4.0) & (df["attack"] == "none")]
    for method in ["coherent_fixed", "adaptive"]:
        sub = far[far["method"] == method]
        if len(sub) > 0:
            lines.append(f"  {method:<20} avg SNR = {sub['snr'].mean():.2f} dB")

    lines.append("")
    lines.append("--- Distance Effect ---")
    for d in DISTANCES:
        sub = df[(df["distance"] == d) & (df["attack"] == "none")]
        avg_snr = sub["snr"].mean()
        lines.append(f"  {d}m: avg SNR = {avg_snr:.2f} dB")

    with open(path, "w") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="MicFrozen Simulation — Speech Privacy Protection"
    )
    parser.add_argument("--single", action="store_true",
                        help="Run single-sample demo only")
    parser.add_argument("--load", type=str, default=None,
                        help="Load specific audio file for demo")
    parser.add_argument("--distance", type=float, default=2.0,
                        help="Eavesdropper distance for demo (m)")
    parser.add_argument("--angle", type=float, default=15.0,
                        help="Angle offset for demo (deg)")
    parser.add_argument("--output", type=str, default="results",
                        help="Output directory")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Audio data directory")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    np.random.seed(args.seed)

    print("=" * 60)
    print("  MicFrozen Simulation — Speech Privacy Protection")
    print("  Gao et al., MobiCom 2023")
    print("=" * 60)

    # Load signal
    print("\n[1/3] Loading test signal...")
    if args.load:
        s, fs = load_specific_file(args.load)
        print(f"      Loaded: {args.load}")
    else:
        s, fs = load_or_generate_test_signal(args.data_dir)
    print(f"      {len(s)/fs:.2f}s @ {fs}Hz  |  peak={np.max(np.abs(s)):.3f}")

    if args.single:
        print("\n[2/3] Running single-sample demo...")
        run_single_demo(s, args.distance, args.angle, args.output)
    else:
        print("\n[2/3] Running batch experiments...")
        df = run_batch_experiments(s, args.output)
        print("\n[3/3] Generating summary charts...")
        generate_summary_charts(df, args.output)

    print(f"\nDone. Results in: {os.path.abspath(args.output)}/")


if __name__ == "__main__":
    main()
