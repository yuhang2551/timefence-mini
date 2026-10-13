import pandas as pd
import pytest

from timefence.core.types import Window
from timefence.data.synthetic import synthetic_store
from timefence.data.universe import liquidity_universe
from timefence.factors.evaluate import ResearchContext


@pytest.fixture(scope="session")
def store():
    """Synthetic market with a clear short-term reversal planted in it."""
    return synthetic_store(n_stocks=60, n_days=600, reversal=0.3, seed=7)


@pytest.fixture(scope="session")
def universe(store):
    return liquidity_universe(store, top_n=40, min_listed_days=365)


@pytest.fixture(scope="session")
def ctx(store, universe):
    return ResearchContext(store, universe, horizon=5)


@pytest.fixture(scope="session")
def window(store):
    return Window(store.calendar[60], store.calendar[-1])


def frames_equal(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    """Same values (NaN equal to NaN) on the same labels, allowing float noise."""
    if not a.index.equals(b.index) or not a.columns.equals(b.columns):
        return False
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-9, atol=1e-12, check_freq=False)
    return True
