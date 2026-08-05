import io

import numpy as np
import pandas as pd
import pytest

from modules.config import CHANNELS, SAMPLING_RATE


@pytest.fixture
def vibration_df() -> pd.DataFrame:
    """A well-formed 3-channel vibration signal."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({ch: rng.normal(0, 0.5, 5000) for ch in CHANNELS})


@pytest.fixture
def vibration_csv(vibration_df) -> io.BytesIO:
    buf = io.BytesIO()
    vibration_df.to_csv(buf, index=False)
    buf.seek(0)
    return buf


@pytest.fixture
def sine_signal() -> np.ndarray:
    """1 kHz sine, amplitude 2.0 — RMS is analytically 2/sqrt(2)."""
    t = np.arange(SAMPLING_RATE) / SAMPLING_RATE
    return 2.0 * np.sin(2 * np.pi * 1000 * t)
