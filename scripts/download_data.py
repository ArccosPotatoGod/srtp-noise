#!/usr/bin/env python3
"""Download speech datasets for MicFrozen simulation."""

import argparse
import subprocess
import sys
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def download_librispeech_sample():
    """Download a small LibriSpeech test-clean sample."""
    dest = DATA_DIR / "LibriSpeech"
    if dest.exists():
        print(f"LibriSpeech already exists at {dest}")
        return
    dest.mkdir(parents=True, exist_ok=True)
    url = "https://www.openslr.org/resources/12/test-clean.tar.gz"
    print(f"Downloading LibriSpeech test-clean from {url} ...")
    subprocess.run(
        ["wget", "-q", "--show-progress", "-O", str(dest / "test-clean.tar.gz"), url],
        check=True,
    )
    subprocess.run(
        ["tar", "-xzf", str(dest / "test-clean.tar.gz"), "-C", str(dest)],
        check=True,
    )
    print("LibriSpeech downloaded and extracted.")


def download_audiomnist_sample():
    """Download AudioMNIST (small subset)."""
    dest = DATA_DIR / "AudioMNIST"
    if dest.exists():
        print(f"AudioMNIST already exists at {dest}")
        return
    # Placeholder — real AudioMNIST requires manual download from GitHub
    print("AudioMNIST requires manual download. See:")
    print("  https://github.com/soerenab/AudioMNIST")


def generate_sample_wav():
    """Generate a minimal sample WAV file for testing."""
    import numpy as np
    import soundfile as sf
    dest = DATA_DIR / "sample.wav"
    if dest.exists():
        return
    fs = 16000
    t = np.arange(fs * 2) / fs  # 2 seconds
    tone = 0.5 * np.sin(2.0 * np.pi * 200.0 * t).astype(np.float32)
    sf.write(dest, tone, fs)
    print(f"Sample WAV generated at {dest}")

    txt_dest = DATA_DIR / "sample.txt"
    txt_dest.write_text("hello world")
    print(f"Sample text generated at {txt_dest}")


def main():
    parser = argparse.ArgumentParser(description="Download datasets for MicFrozen")
    parser.add_argument("--librispeech", action="store_true", help="Download LibriSpeech test-clean")
    parser.add_argument("--audiomnist", action="store_true", help="Download AudioMNIST")
    parser.add_argument("--sample", action="store_true", help="Generate a minimal test WAV")
    parser.add_argument("--all", action="store_true", help="Download/generate everything")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.all or args.sample:
        generate_sample_wav()
    if args.all or args.librispeech:
        download_librispeech_sample()
    if args.all or args.audiomnist:
        download_audiomnist_sample()


if __name__ == "__main__":
    main()
