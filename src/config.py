"""Shared experiment configuration.

Single source of truth for all constants shared between the simulation
pipeline and report generation. Imported by demo_main.py and src/report.py.
"""

FS = 16000
SPEED_OF_SOUND = 340.0

SRC_POS = (0.0, 0.0)
JAMMER_POS = (0.2, 0.0)

DISTANCES = [1.0, 2.0, 3.0, 4.0, 5.0]
ANGLES = [0, 15, 30, 45]
METHODS = ["gaussian", "coherent_fixed", "adaptive"]
ATTACKS = ["none", "bandstop", "bandpass", "ica", "beamforming"]

METHOD_LABELS = {
    "gaussian": "Gaussian (UMJ)",
    "coherent_fixed": "Coherent Fixed (MicFrozen)",
    "adaptive": "Adaptive Coherent (Ours)",
}

ATTACK_LABELS = {
    "none": "No Attack",
    "bandstop": "Bandstop Filter",
    "bandpass": "Bandpass Filter",
    "ica": "ICA (FastICA)",
    "beamforming": "Beamforming",
}
