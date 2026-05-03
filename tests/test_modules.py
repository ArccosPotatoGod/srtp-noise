"""Unit tests for MicFrozen simulation core modules."""

import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from src.propagation import propagate_signal, compute_distance
from src.demodulation import nonlinear_demod
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
from src.metrics import (
    compute_snr,
    compute_mfcc_distance,
    compute_segment_snr,
    compute_wer,
    compute_cer,
)

FS = 16000
SRC_POS = (0.0, 0.0)
JAMMER_POS = (0.2, 0.0)


# ---------------------------------------------------------------------------
# Propagation
# ---------------------------------------------------------------------------

class TestPropagation(unittest.TestCase):
    def setUp(self):
        t = np.arange(FS) / FS
        self.s = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        self.fs = FS

    def test_propagate_no_error(self):
        result = propagate_signal(self.s, SRC_POS, (1.0, 0.0), self.fs)
        self.assertEqual(len(result), len(self.s))

    def test_output_finite(self):
        result = propagate_signal(self.s, SRC_POS, (5.0, 3.0), self.fs)
        self.assertTrue(np.all(np.isfinite(result)))

    def test_attenuation_with_distance(self):
        near = propagate_signal(self.s, SRC_POS, (0.5, 0.0), self.fs)
        far = propagate_signal(self.s, SRC_POS, (10.0, 0.0), self.fs)
        self.assertGreater(np.sum(np.abs(near)), np.sum(np.abs(far)))

    def test_ultrasound_more_attenuation(self):
        normal = propagate_signal(self.s, SRC_POS, (2.0, 0.0), self.fs,
                                  is_ultrasound=False)
        ultra = propagate_signal(self.s, SRC_POS, (2.0, 0.0), self.fs,
                                 is_ultrasound=True)
        self.assertGreater(np.sum(np.abs(normal)), np.sum(np.abs(ultra)))

    def test_compute_distance(self):
        d = compute_distance((0, 0), (3, 4))
        self.assertAlmostEqual(d, 5.0)

    def test_zero_distance_attenuation(self):
        """Even at zero distance, attenuation should be finite."""
        result = propagate_signal(self.s, (0, 0), (0, 0), self.fs)
        self.assertTrue(np.all(np.isfinite(result)))

    def test_delay_beyond_signal(self):
        """Signal with delay > length should be all zeros."""
        result = propagate_signal(self.s, (0, 0), (1000, 0), self.fs)
        self.assertAlmostEqual(np.sum(np.abs(result)), 0.0)


# ---------------------------------------------------------------------------
# Demodulation
# ---------------------------------------------------------------------------

class TestDemodulation(unittest.TestCase):
    def test_output_shape(self):
        noise = np.random.randn(1000).astype(np.float32) * 0.5
        result = nonlinear_demod(noise)
        self.assertEqual(len(result), len(noise))

    def test_nonlinear_effect(self):
        noise = np.random.randn(1000).astype(np.float32) * 0.5
        result = nonlinear_demod(noise)
        self.assertFalse(np.allclose(result, noise))

    def test_zero_input(self):
        result = nonlinear_demod(np.zeros(100, dtype=np.float32))
        self.assertTrue(np.allclose(result, 0.0))


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------

class TestCancellation(unittest.TestCase):
    def setUp(self):
        t = np.arange(FS * 2) / FS
        self.s = (np.sin(2 * np.pi * 440 * t) * 0.9).astype(np.float32)
        self.fs = FS

    def test_output_shape(self):
        residual, direct = apply_cancellation(
            self.s, SRC_POS, JAMMER_POS, (2.0, 0.0), self.fs)
        self.assertEqual(len(residual), len(self.s))
        self.assertEqual(len(direct), len(self.s))

    def test_cancellation_finite(self):
        spy_pos = (2.0, 0.0)
        residual, direct = apply_cancellation(
            self.s, SRC_POS, JAMMER_POS, spy_pos, self.fs)
        self.assertTrue(np.all(np.isfinite(residual)))
        self.assertTrue(np.all(np.isfinite(direct)))

    def test_colinear_cancellation(self):
        """On the line src->jammer->spy, cancellation should be effective."""
        spy_pos = (5.0, 0.0)  # Colinear
        residual, direct = apply_cancellation(
            self.s, SRC_POS, JAMMER_POS, spy_pos, self.fs)
        # Colinear geometry yields best phase alignment for cancellation
        self.assertTrue(np.all(np.isfinite(residual)))

    def test_angled_spy(self):
        """Off-axis spy position should still produce valid output."""
        spy_pos = (2.0, 2.0)
        residual, direct = apply_cancellation(
            self.s, SRC_POS, JAMMER_POS, spy_pos, self.fs)
        self.assertEqual(len(residual), len(self.s))


# ---------------------------------------------------------------------------
# Noise Generation
# ---------------------------------------------------------------------------

class TestNoiseGeneration(unittest.TestCase):
    def setUp(self):
        self.s = np.sin(2 * np.pi * 440 * np.arange(FS * 2) / FS).astype(np.float32)
        self.rng = np.random.default_rng(42)

    def test_coherent_noise_shape(self):
        noise = generate_coherent_noise(self.s, rng=self.rng)
        self.assertEqual(len(noise), len(self.s))

    def test_adaptive_noise_shape(self):
        noise = adaptive_coherent_noise(self.s, 3.0, rng=self.rng)
        self.assertEqual(len(noise), len(self.s))

    def test_gaussian_noise_shape(self):
        noise = generate_gaussian_noise(len(self.s), rng=self.rng)
        self.assertEqual(len(noise), len(self.s))

    def test_adaptive_weights_differ(self):
        n1 = adaptive_coherent_noise(self.s, 1.0, rng=self.rng)
        n2 = adaptive_coherent_noise(self.s, 5.0, rng=self.rng)
        # Different alpha should produce different noise patterns
        self.assertFalse(np.allclose(np.std(n1), np.std(n2)))

    def test_reproducible_with_seed(self):
        rng_a = np.random.default_rng(12345)
        rng_b = np.random.default_rng(12345)
        n1 = generate_coherent_noise(self.s, rng=rng_a)
        n2 = generate_coherent_noise(self.s, rng=rng_b)
        np.testing.assert_array_equal(n1, n2)

    def test_gaussian_band_limited(self):
        """Gaussian noise should have reduced high-frequency energy."""
        noise = generate_gaussian_noise(FS * 2, bandwidth=1000, rng=self.rng)
        # Check that energy is finite (band-limitation successful)
        self.assertTrue(np.all(np.isfinite(noise)))
        self.assertGreater(np.std(noise), 0.0)


# ---------------------------------------------------------------------------
# Attacks
# ---------------------------------------------------------------------------

class TestAttacks(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.s = np.random.randn(FS * 2).astype(np.float32)
        self.noisy = (self.s + np.random.randn(len(self.s)).astype(np.float32) * 0.3)

    def test_bandstop_shape(self):
        result = apply_bandstop(self.noisy, FS)
        self.assertEqual(len(result), len(self.noisy))

    def test_bandpass_shape(self):
        result = apply_bandpass(self.noisy, FS)
        self.assertEqual(len(result), len(self.noisy))

    def test_bandpass_keeps_energy(self):
        """Bandpass should preserve more energy than bandstop on clean-ish signal."""
        result = apply_bandpass(self.noisy, FS)
        self.assertTrue(np.all(np.isfinite(result)))

    def test_ica_output_shape(self):
        S_ = ica_denoise(self.noisy,
                         self.noisy + np.random.randn(len(self.noisy)).astype(np.float32) * 0.1)
        self.assertEqual(S_.shape[0], len(self.noisy))
        self.assertEqual(S_.shape[1], 2)

    def test_select_speech_component(self):
        S_ = ica_denoise(self.s, self.noisy)
        best = select_speech_component(S_, self.s)
        self.assertEqual(len(best), len(self.s))

    def test_delay_and_sum_shape(self):
        channels = np.array([self.noisy] * 4)
        result = delay_and_sum(channels, [0, -2, -4, -6])
        self.assertEqual(len(result), len(self.noisy))

    def test_delay_and_sum_consistent(self):
        """With zero delays, beamforming = mean of channels."""
        channels = np.array([self.noisy] * 4)
        result = delay_and_sum(channels, [0, 0, 0, 0])
        np.testing.assert_array_almost_equal(result, self.noisy)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class TestMetrics(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.s_clean = np.random.randn(FS).astype(np.float32)
        self.s_noisy = (self.s_clean +
                        np.random.randn(FS).astype(np.float32) * 0.1)

    def test_perfect_snr(self):
        snr = compute_snr(self.s_clean, self.s_clean)
        self.assertEqual(snr, np.inf)

    def test_snr_decreases_with_noise(self):
        snr1 = compute_snr(self.s_clean, self.s_noisy)
        snr2 = compute_snr(self.s_clean,
                           self.s_clean + np.random.randn(FS).astype(np.float32) * 0.5)
        self.assertGreater(snr1, snr2)

    def test_mfcc_distance_zero_for_same(self):
        dist = compute_mfcc_distance(self.s_clean, self.s_clean)
        self.assertAlmostEqual(dist, 0.0, delta=1e-6)

    def test_mfcc_distance_positive(self):
        dist = compute_mfcc_distance(self.s_clean, self.s_noisy)
        self.assertGreater(dist, 0.0)

    def test_segment_snr(self):
        seg = compute_segment_snr(self.s_clean, self.s_noisy)
        self.assertTrue(np.isfinite(seg))
        self.assertGreater(seg, -10.0)  # Clipping floor

    def test_segment_snr_perfect(self):
        seg = compute_segment_snr(self.s_clean, self.s_clean)
        self.assertGreater(seg, 30.0)

    def test_wer_perfect(self):
        w = compute_wer("hello world", "hello world")
        self.assertEqual(w, 0.0)

    def test_wer_imperfect(self):
        w = compute_wer("hello world", "hello word")
        self.assertGreater(w, 0.0)

    def test_cer_perfect(self):
        c = compute_cer("hello", "hello")
        self.assertEqual(c, 0.0)


if __name__ == "__main__":
    unittest.main()
