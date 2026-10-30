import io

import numpy as np
import pytest
from scipy.io import wavfile

from rnmp.doa.music import (
    direction_vectors,
    find_peaks_2d,
    mic_positions,
    music_spectrum,
    read_wav,
    spatial_covariances,
)


def plane_wave(azimuth_deg, elevation_deg, sample_rate=48000, samples_count=48000, seed=0):
    """White noise from one direction on the 4 mics (radius 20 mm), fractional delays in the frequency domain."""
    rng = np.random.default_rng(seed)
    positions = mic_positions([0, 1, 2, 3], 0.02)
    u = direction_vectors(np.array([azimuth_deg]), np.array([elevation_deg]))[0, 0]
    spectrum = np.fft.rfft(rng.standard_normal(samples_count))
    freqs = np.fft.rfftfreq(samples_count, 1 / sample_rate)
    x = np.stack([np.fft.irfft(spectrum * np.exp(2j * np.pi * freqs * (p @ u) / 343.0), samples_count)
                  for p in positions], axis=1)
    return x + 0.05 * rng.standard_normal(x.shape), positions


@pytest.mark.parametrize("azimuth_deg, elevation_deg", [(0, 30), (60, 30), (-120, 10), (150, 60)])
def test_music_finds_plane_wave_direction(azimuth_deg, elevation_deg):
    x, positions = plane_wave(azimuth_deg, elevation_deg)
    freqs, R = spatial_covariances(x, 48000, 1024, 500, 4000)
    azimuths, elevations = np.arange(-180, 180), np.arange(0, 91)
    spectrum, _ = music_spectrum(R, freqs, positions, direction_vectors(azimuths, elevations), 1)
    el_idx, az_idx = find_peaks_2d(spectrum, 1)
    assert azimuths[az_idx[0]] == azimuth_deg
    assert elevations[el_idx[0]] == elevation_deg


def test_read_wav_scales_and_names_channels():
    buffer = io.BytesIO()
    wavfile.write(buffer, 48000, np.array([[16384, -32768], [0, 32767]], dtype=np.int16))
    sample_rate, samples, names, is_raw_pdm = read_wav(buffer.getvalue())
    assert sample_rate == 48000
    assert np.allclose(samples, [[0.5, -1.0], [0.0, 32767 / 32768]])
    assert names == ["ch0", "ch1"]
    assert not is_raw_pdm
