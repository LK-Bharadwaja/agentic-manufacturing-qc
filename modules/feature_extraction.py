import logging
import warnings

import numpy as np
import pandas as pd
import scipy.signal as sp_signal
from scipy.fftpack import fft
from scipy.stats import entropy, kurtosis

from modules.config import (
    CHANNELS,
    OVERLAP,
    SAMPLING_RATE,
    STFT_WINDOW,
    WINDOW_SIZE,
)
from modules.exceptions import InvalidInputError, MissingChannelError

logger = logging.getLogger(__name__)

__all__ = [
    "CHANNELS",
    "FEATURE_NAMES",
    "OVERLAP",
    "SAMPLING_RATE",
    "STFT_WINDOW",
    "WINDOW_SIZE",
    "available_channels",
    "extract_features_from_df",
    "extract_features_from_file",
]

FEATURE_NAMES = []
for _ch in range(1, 4):
    FEATURE_NAMES.extend([
        f"Channel {_ch} RMS",
        f"Channel {_ch} PEAK",
        f"Channel {_ch} CREST FACTOR",
        f"Channel {_ch} KURTOSIS",
        f"Channel {_ch} PSD",
        f"Channel {_ch} Dominant Frequency",
        f"Channel {_ch} Total Energy",
        f"Channel {_ch} Spectral Entropy",
        f"Channel {_ch} STFT Centroid",
        f"Channel {_ch} STFT Bandwidth",
        f"Channel {_ch} STFT Entropy",
        f"Channel {_ch} STFT Kurtosis",
    ])


def _time_domain(signal: np.ndarray):
    rms = float(np.sqrt(np.mean(signal ** 2)))
    peak = float(np.max(np.abs(signal)))
    crest = peak / rms if rms > 1e-12 else 0.0
    kurt = float(kurtosis(signal, fisher=True))
    return rms, peak, crest, kurt


def _frequency_domain(signal: np.ndarray):
    n = len(signal)
    step = WINDOW_SIZE
    psd_vals, freq_vals, energy_vals, ent_vals = [], [], [], []

    for start in range(0, n - WINDOW_SIZE + 1, step):
        seg = signal[start: start + WINDOW_SIZE]
        fft_mag = np.abs(fft(seg))
        freqs = np.fft.fftfreq(WINDOW_SIZE, d=1.0 / SAMPLING_RATE)
        psd_vals.append(float(np.mean(fft_mag ** 2)))
        freq_vals.append(float(abs(freqs[np.argmax(fft_mag)])))
        energy_vals.append(float(np.sum(fft_mag ** 2)))
        total = np.sum(fft_mag)
        norm = fft_mag / total if total > 1e-12 else fft_mag
        ent_vals.append(float(entropy(np.abs(norm) + 1e-12)))

    if not psd_vals:
        return 0.0, 0.0, 0.0, 0.0

    return (float(np.mean(psd_vals)), float(np.mean(freq_vals)),
            float(np.mean(energy_vals)), float(np.mean(ent_vals)))


def _stft_domain(signal: np.ndarray):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f, _t, Zxx = sp_signal.stft(signal, fs=SAMPLING_RATE, nperseg=STFT_WINDOW)
        mag = np.abs(Zxx) + 1e-12
        col_sum = np.sum(mag, axis=0)
        centroid = np.sum(f[:, None] * mag, axis=0) / col_sum
        bandwidth = np.sqrt(
            np.sum((f[:, None] - centroid) ** 2 * mag, axis=0) / col_sum
        )
        ent_vals = entropy(mag, axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kurt_vals = kurtosis(mag, axis=0, fisher=False)
        return (
            float(np.nanmean(centroid)),
            float(np.nanmean(bandwidth)),
            float(np.nanmean(ent_vals)),
            float(np.nanmean(kurt_vals)),
        )
    except Exception:
        return 0.0, 0.0, 0.0, 0.0


def _safe_signal(df: pd.DataFrame, col: str) -> np.ndarray:
    """Extract and validate a signal column; pad if too short."""
    if col not in df.columns:
        return None
    sig = df[col].dropna().values.astype(float)
    if len(sig) == 0:
        return None
    if len(sig) < WINDOW_SIZE:
        # Reflect-pad so windowed analysis can proceed
        pad = WINDOW_SIZE - len(sig)
        sig = np.pad(sig, (0, pad), mode='reflect' if len(sig) > 1 else 'constant')
    return sig


def _zero_row(ch_num: int, reason: str) -> dict:
    """Return a dict of zeros for all features of a channel, with a note."""
    row = {}
    for feat in ["RMS", "PEAK", "CREST FACTOR", "KURTOSIS", "PSD",
                 "Dominant Frequency", "Total Energy", "Spectral Entropy",
                 "STFT Centroid", "STFT Bandwidth", "STFT Entropy", "STFT Kurtosis"]:
        row[f"Channel {ch_num} {feat}"] = 0.0
    logger.warning("Channel %d (%s) — filled with zeros.", ch_num, reason)
    return row


def _extract_channel(df: pd.DataFrame, ch_idx: int) -> dict:
    ch_num = ch_idx + 1
    col = CHANNELS[ch_idx]
    sig = _safe_signal(df, col)
    if sig is None:
        return _zero_row(ch_num, f"column '{col}' missing or empty")

    rms, peak, crest, kurt = _time_domain(sig)
    psd, dom_freq, energy, sp_ent = _frequency_domain(sig)
    stft_cen, stft_bw, stft_ent, stft_kurt = _stft_domain(sig)

    return {
        f"Channel {ch_num} RMS": rms,
        f"Channel {ch_num} PEAK": peak,
        f"Channel {ch_num} CREST FACTOR": crest,
        f"Channel {ch_num} KURTOSIS": kurt,
        f"Channel {ch_num} PSD": psd,
        f"Channel {ch_num} Dominant Frequency": dom_freq,
        f"Channel {ch_num} Total Energy": energy,
        f"Channel {ch_num} Spectral Entropy": sp_ent,
        f"Channel {ch_num} STFT Centroid": stft_cen,
        f"Channel {ch_num} STFT Bandwidth": stft_bw,
        f"Channel {ch_num} STFT Entropy": stft_ent,
        f"Channel {ch_num} STFT Kurtosis": stft_kurt,
    }


def available_channels(df: pd.DataFrame) -> list:
    """Return the expected channel columns actually present in the DataFrame."""
    return [ch for ch in CHANNELS if ch in df.columns]


def extract_features_from_df(df: pd.DataFrame) -> pd.DataFrame:
    """Extract 36 vibration features from an already-loaded DataFrame.

    Channels that are absent are zero-filled, but a file with no recognised
    channel at all is rejected rather than silently scored on all-zero input.
    """
    if not available_channels(df):
        raise MissingChannelError(CHANNELS, list(df.columns))

    row = {}
    for ch_idx in range(3):
        row.update(_extract_channel(df, ch_idx))
    return pd.DataFrame([row], columns=FEATURE_NAMES)


def extract_features_from_file(file_path: str) -> pd.DataFrame:
    """Load a .xlsx/.xls/.csv part file and extract 36 features."""
    path = str(file_path)
    if path.lower().endswith('.csv'):
        df = pd.read_csv(path)
    elif path.lower().endswith(('.xlsx', '.xls')):
        df = pd.read_excel(path)
    else:
        raise InvalidInputError(
            f"Unsupported file format: '{path}'. Expected .xlsx, .xls, or .csv."
        )
    return extract_features_from_df(df)
