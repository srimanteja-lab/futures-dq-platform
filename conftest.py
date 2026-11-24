import os
import sys

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


def make_bars(contract="ESZ25", start="2025-11-10 09:00", n=30, price=100.0, volume=10, step=0.25):
    """Continuous, valid 1-minute bars with a gentle up-trend."""
    ts = pd.date_range(start, periods=n, freq="1min")
    closes = [price + i * step for i in range(n)]
    return pd.DataFrame({
        "contract": contract,
        "timestamp": ts,
        "open": [c - step / 2 for c in closes],
        "high": [c + step for c in closes],
        "low": [c - step for c in closes],
        "close": closes,
        "volume": float(volume),
    })


@pytest.fixture
def bars():
    return make_bars()


@pytest.fixture
def root():
    return ROOT
