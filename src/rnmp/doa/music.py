"""
Direction of arrival for the 4-mic square array: WAV loading, geometry and broadband MUSIC.

Used by the marimo notebooks of MicrophoneArray.Modeling: microphone_processing/music.py (one window in
detail) and microphone_processing/doa_tracking.py (angle vs time).

Conventions:
  - Geometry (see MicrophoneArray.Modeling recordering/analyze_tdoa_rawpdm.py): circle radius R, top view,
    user at the bottom, x -> right, y -> away from the user, z -> up.
  - Azimuth 0 = toward the user (-y), 90 = right (+x), 180 = away; elevation 0 = horizon, 90 = straight up.
    A planar array can't tell up from down, so elevations below the horizon mirror the ones above.
"""

import io
import re
import warnings

import numpy as np
from scipy import ndimage, signal
from scipy.io import wavfile

SPEED_OF_SOUND = 343.0  # m/s

# Channel -> mic coordinates in units of R / sqrt(2):
#   ch3 (MIC4) --- ch2 (MIC3)    far pair
#   ch0 (MIC2) --- ch1 (MIC1)    near pair
MIC_LAYOUT = {0: (-1, -1), 1: (+1, -1), 2: (+1, +1), 3: (-1, +1)}


def read_wav(data: bytes):
    """WAV bytes -> (sample_rate, samples (n, ch) float in [-1, 1], track names, is_raw_pdm)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", wavfile.WavFileWarning)  # iXML chunk
        sample_rate, samples = wavfile.read(io.BytesIO(data))
    if samples.ndim == 1:
        samples = samples[:, None]
    if np.issubdtype(samples.dtype, np.integer):
        samples = samples.astype(np.float64) / (np.iinfo(samples.dtype).max + 1)

    track_list = re.search(rb"<TRACK_LIST>(.*?)</TRACK_LIST>", data, re.S)
    names = re.findall(rb"<NAME>(.*?)</NAME>", track_list.group(1)) if track_list else []
    names = [n.decode("utf-8", "replace") for n in names]
    if len(names) != samples.shape[1]:
        names = [f"ch{i}" for i in range(samples.shape[1])]

    # raw-PDM capture that has not gone through pdm_to_pcm.py: ch0-3 are packed PDM bits, not audio
    is_raw_pdm = b"<MODE>raw-pdm</MODE>" in data and b"<CONVERSION>" not in data
    return sample_rate, samples, names, is_raw_pdm


def mic_positions(channels, radius):
    """Coordinates (M, 3) of the selected channels, m."""
    half = radius / np.sqrt(2)
    return np.array([[MIC_LAYOUT[ch][0] * half, MIC_LAYOUT[ch][1] * half, 0.0] for ch in channels])


def alias_frequency(positions, c=SPEED_OF_SOUND):
    """Frequency above which the wavelength is shorter than twice the largest mic spacing."""
    max_spacing = max(np.linalg.norm(a - b) for a in positions for b in positions)
    return c / (2 * max_spacing)


def direction_vectors(azimuths_deg, elevations_deg):
    """Unit vectors toward the source, (n_el, n_az, 3)."""
    el, az = np.meshgrid(np.deg2rad(elevations_deg), np.deg2rad(azimuths_deg), indexing="ij")
    return np.stack([np.cos(el) * np.sin(az), -np.cos(el) * np.cos(az), np.sin(el)], axis=-1)


def band_stft(samples, sample_rate, nfft, f_min, f_max):
    """STFT of samples (n, M), only the bins in [f_min, f_max]: freqs (F,), frame times (T,), X (M, F, T)."""
    freqs, times, X = signal.stft(samples.T, fs=sample_rate, nperseg=nfft)
    band = (freqs >= f_min) & (freqs <= f_max)
    return freqs[band], times, X[:, band, :]


def covariances(X):
    """Spatial covariance R(f) = E[x(f) x(f)^H] over the frames of X (M, F, T): (F, M, M)."""
    return np.einsum("mft,nft->fmn", X, X.conj()) / X.shape[2]


def spatial_covariances(samples, sample_rate, nfft, f_min, f_max):
    """band_stft + covariances: freqs (F,), R (F, M, M)."""
    freqs, _, X = band_stft(samples, sample_rate, nfft, f_min, f_max)
    return freqs, covariances(X)


def steering_vectors(freqs, positions, directions, c=SPEED_OF_SOUND):
    """(F, D, M) steering vectors for directions (..., 3) flattened to D.

    A mic at p hears a plane wave from u earlier by p.u / c, so a(f) = exp(+j 2 pi f p.u / c)
    (scipy.signal.stft uses exp(-j w t)).
    """
    advance = directions.reshape(-1, 3) @ positions.T / c  # (D, M), s
    return np.exp(2j * np.pi * freqs[:, None, None] * advance[None])


def music_spectrum(R, freqs, positions, directions, n_sources, steering=None, c=SPEED_OF_SOUND):
    """Broadband MUSIC pseudospectrum over a grid of directions.

    Per bin: null spectrum |E_n^H a|^2 / |a|^2 in [0, 1] (E_n - noise subspace); it is averaged over the
    bins and the spectrum is 1 / mean. `steering` - precomputed steering_vectors(freqs, positions, directions)
    for repeated calls on the same grid; otherwise they are computed bin by bin (less memory).

    Returns spectrum shaped like directions[..., 0] and eigenvalues (F, M), descending.
    """
    M = positions.shape[0]
    eigvals, V = np.linalg.eigh(R)  # (F, M) ascending, (F, M, M)
    noise_space = V[:, :, : M - n_sources]
    advance = None if steering is not None else directions.reshape(-1, 3) @ positions.T / c
    null = 0.0
    for k, f in enumerate(freqs):
        A = steering[k] if steering is not None else np.exp(2j * np.pi * f * advance)  # (D, M)
        null = null + np.sum(np.abs(A.conj() @ noise_space[k]) ** 2, axis=1) / M
    spectrum = 1.0 / (null / len(freqs) + 1e-12)
    return spectrum.reshape(directions.shape[:-1]), eigvals[:, ::-1]


def find_peaks_2d(spectrum, count):
    """Indexes (el, az) of the `count` highest local maxima; azimuth wraps around."""
    is_max = spectrum == ndimage.maximum_filter(spectrum, size=5, mode=["nearest", "wrap"])
    el_idx, az_idx = np.nonzero(is_max)
    order = np.argsort(spectrum[el_idx, az_idx])[::-1][:count]
    return el_idx[order], az_idx[order]
