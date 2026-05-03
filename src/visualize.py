"""Visualization utilities for experiment results.

Generates:
- Waveform comparison plots
- Spectrograms
- SNR-vs-distance line charts
- WER bar charts
- Heatmaps (distance vs angle)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_waveforms(signals, labels, fs=16000, title="Waveform Comparison"):
    """Plot multiple waveforms in subplots for comparison.

    Parameters
    ----------
    signals : list of np.ndarray
        List of signals to plot.
    labels : list of str
        Labels for each signal.
    fs : int
        Sample rate (for time axis).
    title : str
        Figure title.
    """
    n = len(signals)
    fig, axes = plt.subplots(n, 1, figsize=(12, 2 * n), sharex=True)
    if n == 1:
        axes = [axes]

    time = np.arange(len(signals[0])) / fs
    for ax, sig, label in zip(axes, signals, labels):
        ax.plot(time, sig, linewidth=0.5)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title)
    plt.tight_layout()
    return fig


def plot_spectrogram(signal, fs=16000, title="Spectrogram", ax=None):
    """Plot a spectrogram of the signal.

    Parameters
    ----------
    signal : np.ndarray
        Input signal.
    fs : int
        Sample rate.
    title : str
        Plot title.
    ax : matplotlib.axes.Axes or None
        Axis to plot on; creates new if None.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    spec, freqs, times, im = ax.specgram(signal, Fs=fs, NFFT=512,
                                         noverlap=256, cmap='inferno')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    return ax


def plot_snr_vs_distance(results_df, output_path=None):
    """Line chart: SNR vs distance for different jamming methods.

    Parameters
    ----------
    results_df : pd.DataFrame
        Columns: distance, method, attack, snr
    output_path : str or None
        Path to save figure.
    """
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=results_df, x='distance', y='snr',
                 hue='method', style='attack', markers=True, ax=ax)
    ax.set_xlabel("Eavesdropper Distance (m)")
    ax.set_ylabel("SNR (dB)")
    ax.set_title("SNR vs Distance by Jamming Method")
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    return fig


def plot_wer_bars(results_df, output_path=None):
    """Bar chart: WER comparison across methods.

    Parameters
    ----------
    results_df : pd.DataFrame
        Columns: method, attack, wer
    output_path : str or None
        Path to save figure.
    """
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=results_df, x='method', y='wer', hue='attack', ax=ax)
    ax.set_xlabel("Jamming Method")
    ax.set_ylabel("Word Error Rate")
    ax.set_title("WER by Jamming Method and Attack")
    ax.tick_params(axis='x', rotation=15)
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    return fig


def plot_heatmap(matrix, distances, angles, title="SNR Heatmap",
                 output_path=None):
    """Heatmap of a metric over distance-angle grid.

    Parameters
    ----------
    matrix : np.ndarray, shape (n_distances, n_angles)
        Metric values.
    distances : list of float
        Distance values (y-axis).
    angles : list of float
        Angle values (x-axis).
    title : str
        Plot title.
    output_path : str or None
        Path to save figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(matrix, xticklabels=angles, yticklabels=distances,
                annot=True, fmt='.1f', cmap='RdYlBu_r', ax=ax)
    ax.set_xlabel("Angle (deg)")
    ax.set_ylabel("Distance (m)")
    ax.set_title(title)
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    return fig
