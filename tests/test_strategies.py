import numpy as np
import pandas as pd
import pytest

from qbrp.data import make_sample_prices
from qbrp.errors import BacktestInputError
from qbrp.strategies import (
    MACrossConfig,
    MomentumConfig,
    buy_and_hold,
    ma_crossover,
    momentum,
)


def test_ma_crossover_warmup_is_flat_and_trend_is_long():
    p = pd.Series(np.arange(1.0, 31.0))  # steadily rising
    pos = ma_crossover(p, MACrossConfig(3, 10))
    assert (pos.iloc[:9] == 0).all()  # slow MA undefined for the first 9 bars
    assert (pos.iloc[9:] == 1).all()


def test_ma_crossover_falling_prices_go_short_or_flat():
    p = pd.Series(np.arange(30.0, 0.0, -1.0))
    assert (ma_crossover(p, MACrossConfig(3, 10)).iloc[9:] == -1).all()
    assert (ma_crossover(p, MACrossConfig(3, 10, long_only=True)) == 0).all()


def test_ma_crossover_matches_reference_implementation():
    p = make_sample_prices(n=300, seed=8)["close"]
    pos = ma_crossover(p, MACrossConfig(20, 50))
    f, s = p.rolling(20).mean(), p.rolling(50).mean()
    ref = (f > s).astype(int) - (f < s).astype(int)  # the v0.1 definition
    assert (pos == ref).all()


def test_values_are_in_allowed_set():
    p = make_sample_prices(n=500, seed=1)["close"]
    assert set(ma_crossover(p, MACrossConfig()).unique()) <= {-1.0, 0.0, 1.0}
    assert set(ma_crossover(p, MACrossConfig(long_only=True)).unique()) <= {0.0, 1.0}
    assert set(momentum(p, MomentumConfig(60)).unique()) <= {-1.0, 0.0, 1.0}
    assert (buy_and_hold(p) == 1.0).all()


def test_momentum_sign_and_warmup():
    p = pd.Series([10.0, 11, 12, 11, 9, 8, 9, 12])
    pos = momentum(p, MomentumConfig(lookback=2))
    assert pos.tolist() == [0, 0, 1, 0, -1, -1, 0, 1]


@pytest.mark.parametrize(
    "fn",
    [
        lambda p: ma_crossover(p, MACrossConfig(10, 30)),
        lambda p: momentum(p, MomentumConfig(40)),
    ],
)
def test_signals_use_only_past_data(fn):
    p = make_sample_prices(n=300, seed=2)["close"]
    k = 200
    q = p.copy()
    q.iloc[k + 1 :] *= 3.0
    pd.testing.assert_series_equal(fn(p).iloc[: k + 1], fn(q).iloc[: k + 1])


@pytest.mark.parametrize(
    "make, match",
    [
        (lambda: MACrossConfig(50, 20), "shorter than slow"),
        (lambda: MACrossConfig(20, 20), "shorter than slow"),
        (lambda: MACrossConfig(0, 20), "fast"),
        (lambda: MACrossConfig(5.5, 20), "fast"),
        (lambda: MomentumConfig(0), "lookback"),
    ],
)
def test_invalid_configs(make, match):
    with pytest.raises(BacktestInputError, match=match):
        make()


def test_too_few_prices_raises():
    p = pd.Series(np.linspace(100, 110, 30))
    with pytest.raises(BacktestInputError, match="at least slow=50"):
        ma_crossover(p, MACrossConfig(20, 50))
    with pytest.raises(BacktestInputError, match="lookback"):
        momentum(p, MomentumConfig(30))
