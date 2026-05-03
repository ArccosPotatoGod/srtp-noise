#!/usr/bin/env python3
"""Download LibriSpeech test-clean samples for MicFrozen simulation.

Downloads a small subset (5-10 utterances, 10-15s each) from
LibriSpeech test-clean via torchaudio or direct URL.
"""

import os
import sys
import argparse
import numpy as np
import soundfile as sf


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SAMPLE_RATE = 16000
N_SAMPLES = 5
MIN_DURATION = 10.0  # seconds


def download_librispeech_subset(data_dir, n_samples=N_SAMPLES):
    """Download LibriSpeech test-clean samples using librosa.

    Falls back to synthetic test signals if network is unavailable.
    """
    import librosa

    # Try to download actual LibriSpeech samples via pooch
    try:
        import pooch
        print("Downloading LibriSpeech test-clean samples...")

        # Use pooch to fetch a few flac files from OpenSLR
        base_url = "https://www.openslr.org/resources/12/test-clean/"
        # Known short utterances from test-clean (speaker 61, chapter 70968)
        files = [
            "61/70968/61-70968-0000.flac",
            "61/70968/61-70968-0001.flac",
            "61/70968/61-70968-0002.flac",
            "61/70968/61-70968-0003.flac",
            "61/70968/61-70968-0004.flac",
        ]

        fetched = []
        for f in files[:n_samples]:
            url = base_url + f
            local_path = pooch.retrieve(
                url=url,
                known_hash=None,
                path=os.path.join(data_dir, "LibriSpeech"),
                fname=f.replace("/", "_"),
            )
            fetched.append(local_path)
        print(f"Downloaded {len(fetched)} samples.")
        return fetched

    except Exception as e:
        print(f"Download failed ({e}). Generating synthetic test signals.")
        return generate_synthetic_signals(data_dir, n_samples)


def generate_synthetic_signals(data_dir, n_samples):
    """Generate synthetic speech-like test signals for offline use.

    Uses bandpass-filtered noise with time-varying amplitude envelopes
    to approximate speech. Noise-driven signals have rapidly decaying
    autocorrelation (unlike pure sine waves), avoiding oscillatory SNR
    artifacts when the signal is delayed.
    """
    from scipy.signal import butter, lfilter

    os.makedirs(data_dir, exist_ok=True)
    paths = []

    for i in range(n_samples):
        duration = np.random.uniform(10, 15)
        T = int(SAMPLE_RATE * duration)

        # Speech-band filtered noise (300-3400 Hz) as carrier
        rng = np.random.default_rng(100 + i)
        noise = rng.normal(0, 1, T)

        # Bandpass filter to speech range
        nyq = 0.5 * SAMPLE_RATE
        b, a = butter(4, [300 / nyq, 3400 / nyq], btype="bandpass")
        carrier = lfilter(b, a, noise)

        # Time-varying envelope: syllables (onsets every 150-400ms)
        envelope = np.ones(T)
        syllable_period = int(SAMPLE_RATE * np.random.uniform(0.15, 0.40))
        for onset in range(0, T, syllable_period):
            # Each syllable: fast attack, slow decay
            syl_len = min(syllable_period, T - onset)
            t_syl = np.arange(syl_len) / SAMPLE_RATE
            attack = 0.02 + 0.03 * np.random.random()
            decay = 0.08 + 0.12 * np.random.random()
            amp = 0.3 + 0.7 * np.random.random()
            syl_env = amp * (np.exp(-t_syl / decay) - np.exp(-t_syl / attack))
            syl_env = syl_env / (np.max(np.abs(syl_env)) + 1e-10)
            if onset + syl_len <= T:
                envelope[onset:onset + syl_len] += syl_env

        sig = carrier * envelope

        # Normalize
        peak = np.max(np.abs(sig))
        if peak > 0:
            sig = sig / peak * 0.9

        filepath = os.path.join(data_dir, f"synth_sample_{i:03d}.wav")
        sf.write(filepath, sig.astype(np.float32), SAMPLE_RATE)
        paths.append(filepath)

    print(f"Generated {n_samples} synthetic samples (noise-driven, speech-band).")
    return paths


def main():
    parser = argparse.ArgumentParser(
        description="Download/prepare audio data for MicFrozen simulation"
    )
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES,
                        help=f"Number of samples (default: {N_SAMPLES})")
    parser.add_argument("--no-download", action="store_true",
                        help="Skip download, generate synthetic only")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    if args.no_download:
        paths = generate_synthetic_signals(DATA_DIR, args.n_samples)
    else:
        paths = download_librispeech_subset(DATA_DIR, args.n_samples)

    print(f"\nData files ({len(paths)}):")
    for p in paths:
        info = sf.info(p)
        print(f"  {os.path.basename(p)}: {info.duration:.1f}s, {info.samplerate}Hz")

    # Save file list
    list_path = os.path.join(DATA_DIR, "file_list.txt")
    with open(list_path, "w") as f:
        for p in paths:
            f.write(f"{p}\n")
    print(f"\nFile list saved to {list_path}")


if __name__ == "__main__":
    main()
