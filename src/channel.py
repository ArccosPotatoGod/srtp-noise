"""ChannelModule — RIR computation via Pyroomacoustics with ultrasonic attenuation."""

from typing import Dict, List, Tuple

import numpy as np

SPEED_OF_SOUND = 343.0  # m/s at 20 °C


class UltrasonicAttenuation:
    """Extra air-absorption attenuation for ultrasonic frequencies.

    Models the attenuation difference between audible (< 4 kHz, ~0.001 dB/m)
    and ultrasonic (~40 kHz, ~1.2–1.5 dB/m per ISO 9613-1).
    """

    def __init__(self, carrier_freq: float = 39000.0,
                 temperature: float = 20.0, humidity: float = 50.0):
        self.carrier_freq = carrier_freq
        self.temperature = temperature
        self.humidity = humidity
        self.alpha = self._compute_alpha()

    def _compute_alpha(self) -> float:
        """ISO 9613-1 atmospheric absorption coefficient (Np/m)."""
        T = self.temperature + 273.15
        h = self.humidity
        f = self.carrier_freq
        frO = (24.0 + 4.04e4 * h * (0.02 + h) / (0.391 + h)) * 1e-4
        frN = (T / T) ** (-0.5) * (9.0 + 280.0 * h * np.exp(-4.17 * ((T / T) ** (-1.0 / 3.0) - 1.0)))
        alpha = (1.84e-11 * (T / 293.15) ** 0.5
                 + (T / 293.15) ** (-2.5)
                 * (8.686 * 0.01275 * np.exp(-2239.1 / T) * frO / (frO + f * f / frO)
                    + 8.686 * 0.1068 * np.exp(-3352.0 / T) * frN / (frN + f * f / frN)))
        return alpha * f * f * 1e-3  # dB/m → Np/m

    def apply(self, rir: np.ndarray, distance: float) -> np.ndarray:
        """Apply exponential ultrasonic attenuation to a RIR.

        Attenuation factor = exp(-alpha * distance).
        """
        factor = np.exp(-self.alpha * distance)
        return rir.astype(np.float32) * factor


class ChannelModule:
    """Computes room impulse responses via Pyroomacoustics.

    Separates output into audible-path RIRs and ultrasonic-path RIRs
    (the latter with extra high-frequency attenuation applied).
    """

    def __init__(self, config):
        self.config = config
        self.room_config = config.room
        self.source_config = config.source
        self.jammer_config = config.jammer
        self.spy_mic_config = config.spy_mic
        self.ultrasonic_config = config.ultrasonic
        self.attenuation = UltrasonicAttenuation(
            carrier_freq=self.ultrasonic_config.carrier_freq,
            temperature=self.room_config.temperature,
            humidity=self.room_config.humidity,
        )

    def compute_rir(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Compute all RIRs and split into audible and ultrasonic dicts.

        Returns
        -------
        audible_rirs : dict
            "src_to_ref" : (rir_len,)
            "src_to_spy" : list of (rir_len,)
        ultrasonic_rirs : dict
            "jammer_to_spy" : list of (rir_len,)  -- ultrasonic attenuation applied
        """
        import pyroomacoustics as pra

        dim = list(self.room_config.dim)
        rt60 = self.room_config.rt60
        e_absorption, max_order = pra.inverse_sabine(rt60, dim)
        room = pra.ShoeBox(dim, fs=self.config.sim.fs,
                           materials=pra.Material(e_absorption), max_order=max_order)

        # Add speech source
        room.add_source(list(self.source_config.pos))

        # Add Jammer speaker (single source for both cancel + coherent noise)
        room.add_source(list(self.jammer_config.pos_spk))

        # Add reference microphone
        room.add_microphone(list(self.jammer_config.ref_mic_pos))

        # Add spy microphones
        spy_positions = []
        for pos in self.spy_mic_config.positions:
            room.add_microphone(list(pos))
            spy_positions.append(pos)

        room.compute_rir()

        # Extract RIRs (shape: (n_mics, n_sources, rir_len))
        all_rirs = room.rir

        # Microphone indices: 0=ref_mic, 1..N=spy_mics
        # Source indices: 0=speech, 1=jammer
        src_to_ref = all_rirs[0][0].astype(np.float32)

        n_spy = len(spy_positions)
        src_to_spy = [all_rirs[i + 1][0].astype(np.float32) for i in range(n_spy)]

        # Ultrasonic-path RIRs (jammer → spy), with extra attenuation
        jammer_to_spy = []
        for i, pos in enumerate(spy_positions):
            rir = all_rirs[i + 1][1].astype(np.float32)
            distance = np.linalg.norm(
                np.array(self.jammer_config.pos_spk) - np.array(pos)
            )
            if self.ultrasonic_config.apply_to_rir:
                rir = self.attenuation.apply(rir, distance)
            jammer_to_spy.append(rir)

        audible_rirs = {"src_to_ref": src_to_ref, "src_to_spy": src_to_spy}
        ultrasonic_rirs = {"jammer_to_spy": jammer_to_spy}

        return audible_rirs, ultrasonic_rirs
