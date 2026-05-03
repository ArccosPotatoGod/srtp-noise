# MicFrozen Simulation & Improvement

Simulation reproduction and enhancement of the MicFrozen system described in:

> Gao et al., *"Cancelling Speech Signals for Speech Privacy Protection against Microphone Eavesdropping"*, MobiCom 2023.

## Overview

MicFrozen is a wearable device that protects speech privacy by emitting a combination of:
1. **Cancelling signals** — inverted replicas of the user's speech to neutralize it at eavesdropping microphones.
2. **Coherent noise** — nonlinear noise tightly coupled to the speech signal, resistant to blind source separation (ICA) attacks.

This project simulates the core mechanisms in Python and implements a lightweight improvement: **adaptive-weight coherent noise coupling** based on estimated eavesdropper distance.

## Project Structure

```
srtp-final/
├── README.md
├── requirements.txt
├── .gitignore
├── docs/
│   └── spec.md                  # Detailed experiment design (Chinese)
├── src/
│   ├── __init__.py
│   ├── propagation.py           # Acoustic propagation & attenuation
│   ├── demodulation.py          # Ultrasonic nonlinear demodulation
│   ├── cancellation.py          # Inverse-channel speech cancellation
│   ├── noise.py                 # Coherent noise & Gaussian noise generation
│   ├── attacks.py               # Adversarial denoising (filter, ICA, beamforming)
│   ├── metrics.py               # SNR, WER, evaluation metrics
│   └── visualize.py             # Plotting & visualization
├── data/
│   └── .gitkeep                 # LibriSpeech audio samples
├── results/
│   └── .gitkeep                 # Experiment outputs, CSV, figures
├── tests/
│   ├── __init__.py
│   └── test_modules.py          # Unit tests for core modules
├── scripts/
│   ├── download_data.py         # LibriSpeech data downloader
│   └── download_data.sh
├── demo_main.py                 # Main experiment pipeline
└── paper.pdf                    # Reference paper
```

## Setup

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```bash
# Download sample audio data
python scripts/download_data.py

# Run the full experiment pipeline
python demo_main.py
```

## Key Modules

| Module | Function |
|--------|----------|
| `propagation` | Distance attenuation, delay, ultrasound extra attenuation |
| `demodulation` | Nonlinear demodulation producing audible interference |
| `cancellation` | Inverse-channel speech signal cancellation |
| `noise` | Coherent noise (Eq.17) and Gaussian baseline noise |
| `attacks` | Bandstop filter, FastICA, delay-and-sum beamforming |
| `metrics` | SNR computation, WER approximation via MFCC distance |
| `visualize` | Waveform plots, spectrograms, comparison charts |

## Experiment Matrix

- **Jamming methods**: Gaussian noise, coherent noise (fixed α), adaptive coherent noise
- **Eavesdropper distances**: 1m, 2m, 3m, 4m, 5m
- **Angle offsets** (2D): 0°, 15°, 30°, 45°
- **Denoising attacks**: None, bandstop filter, ICA, beamforming

## Dependencies

- Python 3.9+
- numpy, scipy, librosa, soundfile, matplotlib, seaborn, scikit-learn

## References

- Gao et al., *Cancelling Speech Signals for Speech Privacy Protection against Microphone Eavesdropping*, MobiCom 2023.
