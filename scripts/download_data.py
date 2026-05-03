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

    Uses a combination of sine sweeps and amplitude modulation
    to create simple test signals that approximate speech.
    """
    import scipy.signal as signal

    os.makedirs(data_dir, exist_ok=True)
    paths = []

    for i in range(n_samples):
        duration = np.random.uniform(10, 15)
        t = np.linspace(0, duration, int(SAMPLE_RATE * duration))

        # Speech-like: fundamental + harmonics + noise
        f0 = np.random.uniform(100, 200)
        sig = (
            0.6 * np.sin(2 * np.pi * f0 * t)
            + 0.3 * np.sin(2 * np.pi * f0 * 2 * t)
            + 0.1 * np.sin(2 * np.pi * f0 * 3 * t)
        )

        # Amplitude envelope to simulate syllables
        envelope_freq = np.random.uniform(2, 5)
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * envelope_freq * t)
        sig = sig * envelope

        # Add slight noise
        sig = sig + 0.02 * np.random.randn(len(t))

        # Normalize
        sig = sig / np.max(np.abs(sig)) * 0.9

        filepath = os.path.join(data_dir, f"synth_sample_{i:03d}.wav")
        sf.write(filepath, sig.astype(np.float32), SAMPLE_RATE)
        paths.append(filepath)

    print(f"Generated {n_samples} synthetic samples.")
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
