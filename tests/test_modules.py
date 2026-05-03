"""Unit tests for core MicFrozen simulation modules."""

import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from src.propagation import propagate_signal, compute_distance
from src.demodulation import nonlinear_demod
from src.cancellation import apply_cancellation
from src.noise import generate_coherent_noise, adaptive_coherent_noise, generate_gaussian_noise
from src.attacks import apply_bandstop, ica_denoise, delay_and_sum
from src.metrics import compute_snr, compute_mfcc_distance


FS = 16000
SRC_POS = (0.0, 0.0)
JAMMER_POS = (0.2, 0.0)


class TestPropagation(unittest.TestCase):
    def setUp(self):
        self.s = np.sin(2 * np.pi * 440 * np.arange(FS) / FS).astype(np.float64)
        self.fs = FS

    def test_propagate_no_error(self):
        result = propagate_signal(self.s, SRC_POS, (1.0, 0.0), self.fs)
        self.assertEqual(len(result), len(self.s))

    def test_attenuation_with_distance(self):
        near = propagate_signal(self.s, SRC_POS, (0.5, 0.0), self.fs)
        far = propagate_signal(self.s, SRC_POS, (10.0, 0.0), self.fs)
        # Farther mic should have lower energy
        self.assertGreater(np.sum(np.abs(near)), np.sum(np.abs(far)))

    def test_ultrasound_more_attenuation(self):
        normal = propagate_signal(self.s, SRC_POS, (2.0, 0.0), self.fs, is_ultrasound=False)
        ultra = propagate_signal(self.s, SRC_POS, (2.0, 0.0), self.fs, is_ultrasound=True)
        self.assertGreater(np.sum(np.abs(normal)), np.sum(np.abs(ultra)))

    def test_compute_distance(self):
        d = compute_distance((0, 0), (3, 4))
        self.assertAlmostEqual(d, 5.0)


class TestDemodulation(unittest.TestCase):
    def test_output_shape(self):
        noise = np.random.randn(1000) * 0.5
        result = nonlinear_demod(noise)
        self.assertEqual(len(result), len(noise))

    def test_nonlinear_effect(self):
        """Demodulation should produce output different from input."""
        noise = np.random.randn(1000) * 0.5
        result = nonlinear_demod(noise)
        self.assertFalse(np.allclose(result, noise))


class TestCancellation(unittest.TestCase):
    def setUp(self):
        t = np.arange(FS * 2) / FS
        self.s = (np.sin(2 * np.pi * 440 * t) * 0.9).astype(np.float64)
        self.fs = FS

    def test_output_shape(self):
        residual, direct = apply_cancellation(
            self.s, SRC_POS, JAMMER_POS, (2.0, 0.0), self.fs
        )
        self.assertEqual(len(residual), len(self.s))
        self.assertEqual(len(direct), len(self.s))

    def test_cancellation_produces_finite_output(self):
        """Cancellation output should be finite (not NaN/inf)."""
        spy_pos = (2.0, 0.0)
        residual, direct = apply_cancellation(
            self.s, SRC_POS, JAMMER_POS, spy_pos, self.fs
        )
        self.assertTrue(np.all(np.isfinite(residual)))
        self.assertTrue(np.all(np.isfinite(direct)))


class TestNoiseGeneration(unittest.TestCase):
    def setUp(self):
        self.s = np.random.randn(FS * 2).astype(np.float64)

    def test_coherent_noise_shape(self):
        noise = generate_coherent_noise(self.s)
        self.assertEqual(len(noise), len(self.s))

    def test_adaptive_noise_shape(self):
        noise = adaptive_coherent_noise(self.s, 3.0)
        self.assertEqual(len(noise), len(self.s))

    def test_gaussian_noise_shape(self):
        noise = generate_gaussian_noise(len(self.s))
        self.assertEqual(len(noise), len(self.s))

    def test_adaptive_weights_differ(self):
        """Different distances should produce different noise patterns."""
        n1 = adaptive_coherent_noise(self.s, 1.0)
        n2 = adaptive_coherent_noise(self.s, 5.0)
        # They should differ in energy due to different alpha
        self.assertFalse(np.allclose(np.std(n1), np.std(n2)))


class TestAttacks(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.s = np.random.randn(FS * 2).astype(np.float64)
        self.noisy = self.s + np.random.randn(len(self.s)) * 0.3

    def test_bandstop_shape(self):
        result = apply_bandstop(self.noisy, FS)
        self.assertEqual(len(result), len(self.noisy))

    def test_ica_output_shape(self):
        S_ = ica_denoise(self.noisy, self.noisy + np.random.randn(len(self.noisy)) * 0.1)
        # FastICA returns (n_samples, n_components)
        self.assertEqual(S_.shape[0], len(self.noisy))
        self.assertEqual(S_.shape[1], 2)

    def test_delay_and_sum_shape(self):
        channels = np.array([self.noisy] * 4)
        result = delay_and_sum(channels, [0, -2, -4, -6])
        self.assertEqual(len(result), len(self.noisy))


class TestMetrics(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.s_clean = np.random.randn(FS).astype(np.float64)
        self.s_noisy = self.s_clean + np.random.randn(FS) * 0.1

    def test_perfect_snr(self):
        snr = compute_snr(self.s_clean, self.s_clean)
        self.assertEqual(snr, np.inf)

    def test_snr_decreases_with_noise(self):
        snr1 = compute_snr(self.s_clean, self.s_noisy)
        snr2 = compute_snr(self.s_clean, self.s_clean + np.random.randn(FS) * 0.5)
        self.assertGreater(snr1, snr2)

    def test_mfcc_distance_zero_for_same(self):
        dist = compute_mfcc_distance(self.s_clean, self.s_clean)
        self.assertAlmostEqual(dist, 0.0, delta=1e-6)

    def test_mfcc_distance_positive(self):
        dist = compute_mfcc_distance(self.s_clean, self.s_noisy)
        self.assertGreater(dist, 0.0)


if __name__ == "__main__":
    unittest.main()
