"""
Data preprocessing utilities for DAOS-RFF.
Converts raw LoRa IQ samples to STFT spectrograms.

IMPORTANT: Uses only REAL data - no synthetic generation.
"""

import numpy as np
from typing import Tuple, Optional


def compute_stft_spectrogram(
    iq_signal: np.ndarray,
    n_fft: int = 256,
    hop_length: int = 64,
    window: str = "hann",
) -> np.ndarray:
    """
    Compute STFT spectrogram from IQ signal.

    Args:
        iq_signal: Complex IQ samples (N,) or real interleaved (2N,)
        n_fft: FFT size
        hop_length: Hop length between frames
        window: Window function name

    Returns:
        Magnitude spectrogram of shape (n_fft//2+1, num_frames)
    """
    # Handle real interleaved format (I, Q, I, Q, ...)
    if iq_signal.dtype in [np.float32, np.float64] and len(iq_signal.shape) == 1:
        if len(iq_signal) % 2 == 0:
            # Interleaved I/Q - convert to complex
            iq_signal = iq_signal[0::2] + 1j * iq_signal[1::2]

    # Ensure complex format
    if not np.iscomplexobj(iq_signal):
        iq_signal = iq_signal.astype(np.complex64)

    # Create window
    if window == "hann":
        win = np.hanning(n_fft)
    elif window == "hamming":
        win = np.hamming(n_fft)
    else:
        win = np.ones(n_fft)

    # Compute number of frames
    num_samples = len(iq_signal)
    num_frames = 1 + (num_samples - n_fft) // hop_length

    if num_frames <= 0:
        # Pad signal if too short
        pad_length = n_fft - num_samples + hop_length
        iq_signal = np.pad(iq_signal, (0, pad_length), mode='constant')
        num_frames = 1 + (len(iq_signal) - n_fft) // hop_length

    # Compute STFT
    # For complex IQ signals, use fft (not rfft which requires real input)
    stft_matrix = np.zeros((n_fft // 2 + 1, num_frames), dtype=np.float32)

    for frame_idx in range(num_frames):
        start = frame_idx * hop_length
        end = start + n_fft
        frame = iq_signal[start:end] * win
        # Use fft for complex signals, take positive frequencies only
        spectrum = np.fft.fft(frame.astype(np.complex128))
        stft_matrix[:, frame_idx] = np.abs(spectrum[:n_fft // 2 + 1]).astype(np.float32)

    return stft_matrix


def normalize_spectrogram(
    spectrogram: np.ndarray,
    method: str = "minmax"
) -> np.ndarray:
    """
    Normalize spectrogram values.

    Args:
        spectrogram: Input spectrogram
        method: Normalization method ('minmax', 'zscore', 'log')

    Returns:
        Normalized spectrogram
    """
    if method == "minmax":
        min_val = spectrogram.min()
        max_val = spectrogram.max()
        if max_val - min_val > 1e-8:
            return (spectrogram - min_val) / (max_val - min_val)
        return spectrogram - min_val

    elif method == "zscore":
        mean = spectrogram.mean()
        std = spectrogram.std()
        if std > 1e-8:
            return (spectrogram - mean) / std
        return spectrogram - mean

    elif method == "log":
        # Log magnitude with floor to avoid log(0)
        return np.log1p(spectrogram)

    else:
        return spectrogram


def resize_spectrogram(
    spectrogram: np.ndarray,
    target_height: int = 129,
    target_width: int = 64
) -> np.ndarray:
    """
    Resize spectrogram to fixed dimensions using simple interpolation.

    Args:
        spectrogram: Input spectrogram (H, W)
        target_height: Target height
        target_width: Target width

    Returns:
        Resized spectrogram
    """
    from scipy.ndimage import zoom

    h, w = spectrogram.shape
    zoom_h = target_height / h
    zoom_w = target_width / w

    resized = zoom(spectrogram, (zoom_h, zoom_w), order=1)
    return resized.astype(np.float32)
