import math

import numpy as np
import pandas as pd
import pytest

from qbrp.data import make_sample_prices
from qbrp.engine import run_backtest
from qbrp.errors import BacktestInputError
from qbrp.strategies import MACrossConfig, buy_and_hold, ma_crossover

IDX = pd.bdate_range("2024-01-01", periods=5)
PRICES = pd.Series([100.0, 110.0, 99.0, 99.0, 108.9], index=IDX)  # r = 0, +10%, -10%, 0, +10%
TARGET = pd.Series([1.0, 1.0, -1.0, -1.0, 0.0], index=IDX)


def test_hand_computed_same_bar_execution():
    res = run_backtest(PRICES, TARGET, fee_bps=10.0, execution_lag=0)
    assert res.held_position.tolist() == [1, 1, -1, -1, 0]
    assert res.turnover.tolist() == pytest.approx([1, 0, 2, 0, 1])
    assert res.costs.tolist() == pytest.approx([0.001, 0, 0.002, 0, 0.001])
    assert res.gross_returns.tolist() == pytest.approx([0, 0.1, -0.1, 0, -0.1])
    assert res.strategy_returns.tolist() == pytest.approx([-0.001, 0.1, -0.102, 0, -0.101])
    expected_equity = np.cumprod([0.999, 1.1, 0.898, 1.0, 0.899])
    assert res.equity.tolist() == pytest.approx(expected_equity.tolist())
    assert res.total_return == pytest.approx(expected_equity[-1] - 1)
    assert res.n_trades == 3
    assert res.total_turnover == pytest.approx(4.0)
    assert res.total_cost == pytest.approx(0.004)


def test_hand_computed_next_bar_execution():
    res = run_backtest(PRICES, TARGET, fee_bps=10.0, execution_lag=1)
    assert res.held_position.tolist() == [0, 1, 1, -1, -1]
    assert res.strategy_returns.tolist() == pytest.approx([0, -0.001, -0.1, -0.002, -0.1])


def test_default_execution_lag_is_next_bar():
    assert run_backtest(PRICES, TARGET).execution_lag == 1


def test_lag_zero_reproduces_v01_formula():
    df = make_sample_prices(n=300, seed=11)
    pos = ma_crossover(df["close"], MACrossConfig(5, 20))
    res = run_backtest(df["close"], pos, fee_bps=2.5, execution_lag=0)
    r = df["close"].pct_change().fillna(0.0)
    prev = pos.shift(1).fillna(0.0)
    v01 = prev * r - 2.5e-4 * (pos - prev).abs()
    pd.testing.assert_series_equal(res.strategy_returns, v01, check_names=False)


@pytest.mark.parametrize("lag", [0, 1, 2, 5])
def test_lag_k_equals_shifting_the_signal(lag):
    df = make_sample_prices(n=300, seed=3)
    pos = ma_crossover(df["close"], MACrossConfig(5, 20))
    a = run_backtest(df["close"], pos, fee_bps=1.0, execution_lag=lag)
    b = run_backtest(df["close"], pos.shift(lag).fillna(0.0), fee_bps=1.0, execution_lag=0)
    pd.testing.assert_series_equal(a.strategy_returns, b.strategy_returns)


def test_buy_and_hold_without_costs_tracks_the_price():
    df = make_sample_prices(n=250, seed=5)
    res = run_backtest(df["close"], buy_and_hold(df["close"]), fee_bps=0.0, execution_lag=0)
    expected = df["close"] / df["close"].iloc[0]
    np.testing.assert_allclose(res.equity.to_numpy(), expected.to_numpy(), rtol=1e-12)
    assert res.n_trades == 1
    assert res.exposure == 1.0


def test_flat_strategy_earns_nothing_and_sharpe_is_undefined():
    df = make_sample_prices(n=100)
    flat = pd.Series(0.0, index=df.index)
    res = run_backtest(df["close"], flat, fee_bps=50.0)
    assert (res.strategy_returns == 0).all()
    assert res.total_return == 0.0
    assert math.isnan(res.sharpe)
    assert res.n_trades == 0


def test_costs_scale_linearly_with_fee_and_turnover():
    df = make_sample_prices(n=400, seed=9)
    pos = ma_crossover(df["close"], MACrossConfig(3, 10))
    free = run_backtest(df["close"], pos, fee_bps=0.0)
    paid = run_backtest(df["close"], pos, fee_bps=20.0)
    np.testing.assert_allclose(
        (free.strategy_returns - paid.strategy_returns).to_numpy(),
        (20e-4 * paid.turnover).to_numpy(),
        atol=1e-15,
    )
    assert paid.total_return < free.total_return


def test_no_lookahead_future_prices_cannot_change_past_returns():
    """Rewriting prices after bar k must leave strategy returns up to bar k untouched."""
    df = make_sample_prices(n=400, seed=21)
    close = df["close"]
    k = 250
    rng = np.random.default_rng(0)
    tampered = close.copy()
    shocks = rng.normal(0, 0.05, len(close) - k - 1)
    tampered.iloc[k + 1 :] = close.iloc[k] * np.exp(np.cumsum(shocks))
    for lag in (0, 1, 3):
        cfg = MACrossConfig(10, 30)
        a = run_backtest(close, ma_crossover(close, cfg), fee_bps=5.0, execution_lag=lag)
        b = run_backtest(tampered, ma_crossover(tampered, cfg), fee_bps=5.0, execution_lag=lag)
        pd.testing.assert_series_equal(
            a.strategy_returns.iloc[: k + 1], b.strategy_returns.iloc[: k + 1]
        )


def test_peeking_strategy_is_detected_by_lag_zero_vs_true_future():
    """Sanity check: a signal that uses tomorrow's return looks wonderful - which is the point."""
    df = make_sample_prices(n=500, seed=4)
    future_ret = df["close"].pct_change().shift(-1).fillna(0.0)
    cheating = np.sign(future_ret).astype(float)
    res = run_backtest(df["close"], cheating, fee_bps=0.0, execution_lag=0)
    assert res.sharpe > 10  # impossible in reality; the engine faithfully reports the bug


def test_summary_is_complete_and_serialisable():
    import json

    res = run_backtest(PRICES, TARGET, fee_bps=1.0)
    s = res.summary()
    assert {"total_return", "sharpe", "max_drawdown", "cagr", "n_trades", "execution_lag"} <= set(s)
    json.dumps(s)


def test_result_series_share_the_price_index():
    df = make_sample_prices(n=200)
    res = run_backtest(df["close"], ma_crossover(df["close"], MACrossConfig(5, 20)))
    for s in (res.equity, res.strategy_returns, res.held_position, res.costs, res.drawdown):
        assert s.index.equals(df.index)
        assert s.isna().sum() == 0


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"position": TARGET.iloc[:-1]}, "index must match"),
        ({"position": TARGET.reset_index(drop=True)}, "index must match"),
        ({"position": TARGET.where(TARGET > 0)}, "NaN"),
        ({"position": TARGET * 2}, r"\[-1, 1\]"),
        ({"position": TARGET.tolist()}, "pandas Series"),
        ({"fee_bps": -1.0}, "fee_bps"),
        ({"fee_bps": float("nan")}, "fee_bps"),
        ({"execution_lag": -1}, "execution_lag"),
        ({"execution_lag": 1.5}, "execution_lag"),
        ({"execution_lag": 5}, "execution_lag"),
        ({"close": PRICES.where(PRICES > 100, -1.0)}, "strictly positive"),
        ({"close": PRICES.iloc[::-1]}, "index"),
        ({"periods_per_year": 0}, "periods_per_year"),
    ],
)
def test_invalid_inputs_raise(kwargs, match):
    args = {"close": PRICES, "position": TARGET, "fee_bps": 1.0, "execution_lag": 1}
    args.update(kwargs)
    with pytest.raises(BacktestInputError, match=match):
        run_backtest(**args)
