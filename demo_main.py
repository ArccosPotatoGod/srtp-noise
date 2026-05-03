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


def apply_attack(mixed, s_original, attack_name, distance, angle_rad,
                 method_name="coherent_fixed"):
    """Apply a single denoising attack to the mixed signal.

    Parameters
    ----------
    method_name : str
        Jamming method name, used by ICA to match noise type on second channel.
    """
    if attack_name == "none":
        return mixed

    elif attack_name == "bandstop":
        return apply_bandstop(mixed, FS)

    elif attack_name == "bandpass":
        return apply_bandpass(mixed, FS)

    elif attack_name == "ica":
        # Second channel: slightly different spy position, SAME noise type
        spy2 = (distance * np.cos(angle_rad + 0.1),
                distance * np.sin(angle_rad + 0.1))
        res2, _ = apply_cancellation(s_original, SRC_POS, JAMMER_POS, spy2, FS)

        rng2 = np.random.default_rng(42)
        if method_name == "gaussian":
            n2 = generate_gaussian_noise(len(s_original), FS, rng=rng2)
        elif method_name == "adaptive":
            n2 = adaptive_coherent_noise(s_original, distance, FS, rng=rng2)
        else:
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

    # 4. Build metrics for both console and text report
    report_data = {
        "spy_distance": spy_distance, "angle_deg": angle_deg,
        "fs": FS, "src_pos": SRC_POS, "jammer_pos": JAMMER_POS,
        "metrics": [], "signal_stats": [], "noise_stats": [], "geometry": [],
    }

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
        mfcc = compute_mfcc_distance(s, sig, FS)
        report_data["metrics"].append({
            "label": name, "snr": snr, "seg_snr": seg, "mfcc_dist": mfcc,
        })
        print(f"{name:<35} {snr:>10.2f} {seg:>12.2f}")

    for method_name in METHODS:
        label = f"+ {method_name}"
        snr = compute_snr(s, mixed[method_name])
        seg = compute_segment_snr(s, mixed[method_name])
        mfcc = compute_mfcc_distance(s, mixed[method_name], FS)
        report_data["metrics"].append({
            "label": label, "snr": snr, "seg_snr": seg, "mfcc_dist": mfcc,
        })
        print(f"{label:<35} {snr:>10.2f} {seg:>12.2f}")

    # Also show after-attack results (one attack per method)
    print(f"\n{'After denoising attacks:':<35} {'SNR (dB)':>10} {'SegSNR (dB)':>12}")
    print(f"{'-'*57}")
    for method_name in METHODS:
        for att in ["bandstop", "bandpass", "ica", "beamforming"]:
            angle_rad = np.deg2rad(angle_deg)
            processed = apply_attack(mixed[method_name], s, att,
                                     spy_distance, angle_rad,
                                     method_name=method_name)
            snr = compute_snr(s, processed)
            seg = compute_segment_snr(s, processed)
            mfcc = compute_mfcc_distance(s, processed, FS)
            report_data["metrics"].append({
                "label": f"{method_name}→{att}", "snr": snr,
                "seg_snr": seg, "mfcc_dist": mfcc,
            })
            if method_name == "coherent_fixed":
                print(f"  {method_name}→{att:<24} {snr:>10.2f} {seg:>12.2f}")

    # Signal statistics
    for name, sig in [("original s(t)", s), ("direct at spy", s_direct),
                       ("residual", residual)]:
        report_data["signal_stats"].append({
            "label": name,
            "min": float(np.min(sig)), "max": float(np.max(sig)),
            "rms": float(np.sqrt(np.mean(sig ** 2))),
            "peak_to_rms": float(np.max(np.abs(sig)) / (np.sqrt(np.mean(sig ** 2)) + 1e-10)),
        })

    # Noise statistics
    for method_name in METHODS:
        n = noises[method_name][:len(residual)]
        report_data["noise_stats"].append({
            "label": f"{method_name} noise",
            "rms": float(np.sqrt(np.mean(n ** 2))),
            "peak": float(np.max(np.abs(n))),
        })

    # Geometry info
    for label, p in [("src", SRC_POS), ("jammer", JAMMER_POS), ("spy", spy_pos)]:
        d = np.linalg.norm(np.array(p) - np.array(SRC_POS))
        delay = int(round(d / SPEED_OF_SOUND * FS))
        report_data["geometry"].append({
            "label": label, "pos": p, "dist": d, "delay": delay,
        })

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

    # Also add post-ICA on coherent
    angle_rad = np.deg2rad(angle_deg)
    ica_result = apply_attack(mixed["coherent_fixed"], s, "ica",
                              spy_distance, angle_rad,
                              method_name="coherent_fixed")
    spec_signals.append(ica_result)
    spec_labels.append("Coherent → ICA denoised")

    fig_spec = plot_spectrogram_comparison(
        spec_signals, spec_labels, FS,
        title=f"Spectrograms (d={spy_distance}m, {angle_deg}°)",
        output_path=os.path.join(output_dir, "spectrogram_demo.png"))
    plt.close(fig_spec)
    print(f"Spectrogram plot -> {output_dir}/spectrogram_demo.png")

    # 7. Save single-demo text report
    write_single_demo_report(report_data, output_dir)

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
                        mixed, s, attack_name, distance, angle_rad,
                        method_name=method_name)

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


def _fmt_table(headers, rows, col_widths=None):
    """Format aligned text table. Returns list of strings."""
    if col_widths is None:
        col_widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
                      for i, h in enumerate(headers)]
    sep = "  "
    out = []
    out.append(sep.join(str(h).ljust(col_widths[i]) for i, h in enumerate(headers)))
    out.append(sep.join("-" * w for w in col_widths))
    for row in rows:
        out.append(sep.join(str(v).ljust(col_widths[i]) for i, v in enumerate(row)))
    return out


def _pivot_table(df, index_col, column_col, value_col, agg="mean"):
    """Return a formatted pivot table as list of strings."""
    pivot = df.pivot_table(values=value_col, index=index_col,
                           columns=column_col, aggfunc=agg)
    # Build rows: index label + each column value
    headers = [index_col] + [str(c) for c in pivot.columns]
    rows = []
    for idx, row in pivot.iterrows():
        rows.append([str(idx)] + [f"{v:.2f}" for v in row.values])
    return _fmt_table(headers, rows)


def write_summary_report(df, path):
    """Write comprehensive text report with all numerical results."""

    def w(lines_list):
        """Join and write a block of lines."""
        if isinstance(lines_list, list):
            f.write("\n".join(lines_list) + "\n\n")

    with open(path, "w") as f:
        # ── Header ──
        f.write("=" * 78 + "\n")
        f.write("  MicFrozen Simulation — Comprehensive Experiment Report\n")
        f.write("  Gao et al., MobiCom 2023\n")
        f.write("=" * 78 + "\n\n")

        f.write(f"Generated: {pd.Timestamp.now()}\n")
        f.write(f"Rows: {len(df)}\n")
        f.write(f"Distances: {DISTANCES}\n")
        f.write(f"Angles: {ANGLES}\n")
        f.write(f"Methods: {METHODS}\n")
        f.write(f"Attacks: {ATTACKS}\n")
        f.write(f"Metrics: snr, seg_snr, mfcc_distance\n\n")

        # ── Section 1: Full Data Table ──
        f.write("=" * 78 + "\n")
        f.write("  SECTION 1: FULL DATA TABLE (all combinations)\n")
        f.write("=" * 78 + "\n\n")

        cols = ["distance", "angle", "method", "attack", "snr", "seg_snr", "mfcc_distance"]
        headers = ["dist", "ang", "method", "attack", "snr(dB)", "segSNR", "mfcc_dist"]
        rows = []
        for _, r in df.iterrows():
            rows.append([
                f"{r['distance']:.0f}",
                f"{r['angle']:.0f}",
                r["method"],
                r["attack"],
                f"{r['snr']:.2f}",
                f"{r['seg_snr']:.2f}",
                f"{r['mfcc_distance']:.3f}",
            ])
        w(_fmt_table(headers, rows))

        # ── Section 2: SNR Pivot — Method × Attack ──
        f.write("=" * 78 + "\n")
        f.write("  SECTION 2: SNR(dB) — Method × Attack (avg over distance & angle)\n")
        f.write("=" * 78 + "\n\n")
        w(_pivot_table(df, "method", "attack", "snr"))

        # ── Section 3: SNR Pivot — Distance × Method ──
        f.write("=" * 78 + "\n")
        f.write("  SECTION 3: SNR(dB) — Distance × Method (avg over angle, no attack)\n")
        f.write("=" * 78 + "\n\n")
        df_noatt = df[df["attack"] == "none"]
        w(_pivot_table(df_noatt, "distance", "method", "snr"))

        # ── Section 4: SNR Pivot — Distance × Angle ──
        f.write("=" * 78 + "\n")
        f.write("  SECTION 4: SNR(dB) — Distance × Angle (avg over method, no attack)\n")
        f.write("=" * 78 + "\n\n")
        w(_pivot_table(df_noatt, "distance", "angle", "snr"))

        # ── Section 5: SegSNR Pivots ──
        f.write("=" * 78 + "\n")
        f.write("  SECTION 5: SegSNR(dB) — Method × Attack (avg over distance & angle)\n")
        f.write("=" * 78 + "\n\n")
        w(_pivot_table(df, "method", "attack", "seg_snr"))

        # ── Section 6: MFCC Distance Pivots ──
        f.write("=" * 78 + "\n")
        f.write("  SECTION 6: MFCC Distance — Method × Attack (avg, lower=better)\n")
        f.write("=" * 78 + "\n\n")
        w(_pivot_table(df, "method", "attack", "mfcc_distance"))

        # ── Section 7: SNR Heatmap Matrices (text) ──
        f.write("=" * 78 + "\n")
        f.write("  SECTION 7: SNR Heatmap Matrices (distance × angle, no attack)\n")
        f.write("=" * 78 + "\n\n")
        for method in METHODS:
            f.write(f"--- {method} ---\n")
            sub = df[(df["method"] == method) & (df["attack"] == "none")]
            pivot = sub.pivot_table(values="snr", index="distance",
                                    columns="angle", aggfunc="mean")
            # Headers: angle labels
            ang_labels = [f"{int(a)}°" for a in pivot.columns]
            header_line = "dist  " + "  ".join(f"{a:>7}" for a in ang_labels)
            f.write(header_line + "\n")
            f.write("-" * len(header_line) + "\n")
            for dist_idx, row in pivot.iterrows():
                vals = "  ".join(f"{v:>7.2f}" for v in row.values)
                f.write(f"{dist_idx:>4.0f}  {vals}\n")
            f.write("\n")

        # ── Section 8: Statistical Summary ──
        f.write("=" * 78 + "\n")
        f.write("  SECTION 8: Statistical Summary (SNR dB, no attack)\n")
        f.write("=" * 78 + "\n\n")

        for group_col, group_name in [("method", "Method"), ("distance", "Distance"),
                                       ("angle", "Angle")]:
            f.write(f"--- By {group_name} ---\n")
            grouped = df_noatt.groupby(group_col)["snr"]
            stats_rows = []
            for name, grp in grouped:
                stats_rows.append([
                    str(name),
                    f"{grp.mean():.2f}",
                    f"{grp.std():.2f}",
                    f"{grp.min():.2f}",
                    f"{grp.max():.2f}",
                    f"{len(grp)}",
                ])
            w(_fmt_table(
                [group_name, "mean", "std", "min", "max", "N"], stats_rows))
            f.write("\n")

        # ── Section 9: Attack Impact ──
        f.write("=" * 78 + "\n")
        f.write("  SECTION 9: Attack Impact — SNR change relative to no-attack\n")
        f.write("=" * 78 + "\n\n")

        baseline = df_noatt.groupby(["distance", "angle", "method"])["snr"].mean()
        impact_rows = []
        for att in [a for a in ATTACKS if a != "none"]:
            sub = df[df["attack"] == att].copy()
            sub["baseline"] = sub.apply(
                lambda r: baseline.get((r["distance"], r["angle"], r["method"]), 0),
                axis=1)
            sub["delta"] = sub["snr"] - sub["baseline"]
            impact_rows.append([
                att,
                f"{sub['delta'].mean():+.2f}",
                f"{sub['delta'].std():.2f}",
                f"{sub['delta'].min():+.2f}",
                f"{sub['delta'].max():+.2f}",
            ])
        w(_fmt_table(
            ["attack", "delta_mean", "delta_std", "delta_min", "delta_max"], impact_rows))
        f.write("(positive delta = attack improved SNR / recovered speech)\n\n")

        # ── Section 10: Top/Bottom performers ──
        f.write("=" * 78 + "\n")
        f.write("  SECTION 10: Best & Worst Privacy (SNR, lower = better privacy)\n")
        f.write("=" * 78 + "\n\n")

        top10 = df.nsmallest(10, "snr")
        bottom10 = df.nlargest(10, "snr")

        f.write("--- Top 10 Lowest SNR (best privacy) ---\n")
        for _, r in top10.iterrows():
            f.write(f"  d={r['distance']:.0f}m  a={r['angle']:>2}deg  "
                    f"method={r['method']:<15}  attack={r['attack']:<12}  "
                    f"SNR={r['snr']:.2f}dB\n")
        f.write(f"\n--- Top 10 Highest SNR (worst privacy) ---\n")
        for _, r in bottom10.iterrows():
            f.write(f"  d={r['distance']:.0f}m  a={r['angle']:>2}deg  "
                    f"method={r['method']:<15}  attack={r['attack']:<12}  "
                    f"SNR={r['snr']:.2f}dB\n")

        f.write("\n" + "=" * 78 + "\n")
        f.write("  END OF REPORT\n")
        f.write("=" * 78 + "\n")


def write_single_demo_report(data, output_dir):
    """Save single-sample demo results as text report."""
    path = os.path.join(output_dir, "single_demo_report.txt")
    with open(path, "w") as f:
        def w(s):
            f.write(s + "\n")

        w("=" * 70)
        w("  Single-Sample Demo — Detailed Report")
        for key in ["spy_distance", "angle_deg", "fs", "src_pos", "jammer_pos"]:
            if key in data:
                w(f"  {key}: {data[key]}")
        w("=" * 70)
        w("")

        # SNR table
        w(f"{'Signal':<35} {'SNR (dB)':>10} {'SegSNR (dB)':>12} {'MFCC dist':>10}")
        w("-" * 67)
        for entry in data.get("metrics", []):
            w(f"{entry['label']:<35} {entry['snr']:>10.2f} "
              f"{entry['seg_snr']:>12.2f} {entry['mfcc_dist']:>10.3f}")

        w("")
        w("--- Signal Statistics ---")
        for entry in data.get("signal_stats", []):
            w(f"  {entry['label']:<30}  min={entry['min']:+.4f}  "
              f"max={entry['max']:+.4f}  rms={entry['rms']:.4f}  "
              f"peak_to_rms={entry['peak_to_rms']:.1f}")

        w("")
        w("--- Noise Statistics ---")
        for entry in data.get("noise_stats", []):
            w(f"  {entry['label']:<30}  rms={entry['rms']:.4f}  "
              f"peak={entry['peak']:.4f}")

        w("")
        w("--- Eavesdropper Geometry ---")
        for entry in data.get("geometry", []):
            w(f"  {entry['label']:<30}  pos=({entry['pos'][0]:.3f}, {entry['pos'][1]:.3f})"
              f"  dist={entry['dist']:.3f}m  delay={entry['delay']}samples")

    print(f"Single-demo report -> {path}")


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
