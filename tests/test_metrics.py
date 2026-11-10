import math

import numpy as np
import pandas as pd
import pytest

from qbrp.errors import BacktestInputError
from qbrp.metrics import (
    annualised_volatility,
    cagr,
    daily_returns,
    drawdown_series,
    equity_curve,
    expected_max_sharpe_under_null,
    max_drawdown,
    sharpe_ratio,
    sharpe_standard_error,
    simple_returns,
    summarise_returns,
    total_return,
)


def test_simple_returns_hand_computed():
    p = pd.Series([100.0, 110.0, 99.0])
    r = simple_returns(p)
    assert r.tolist() == pytest.approx([0.0, 0.10, -0.10])
    assert r.name == "asset_ret"


def test_daily_returns_alias_kept_for_v01_callers():
    assert daily_returns is simple_returns


def test_equity_curve_compounds():
    eq = equity_curve(pd.Series([0.10, -0.10, 0.0]), start=2.0)
    assert eq.tolist() == pytest.approx([2.2, 1.98, 1.98])


def test_max_drawdown_hand_computed():
    eq = pd.Series([1.0, 1.2, 0.9, 1.1, 1.3, 1.17])
    # Worst decline is 1.2 -> 0.9 = -25%, deeper than 1.3 -> 1.17 = -10%.
    assert max_drawdown(eq) == pytest.approx(-0.25)
    assert drawdown_series(eq).iloc[0] == 0.0


def test_max_drawdown_zero_for_monotone_equity():
    assert max_drawdown(pd.Series([1.0, 1.01, 1.02, 1.5])) == 0.0


def test_max_drawdown_non_positive_on_random_paths():
    rng = np.random.default_rng(0)
    for _ in range(20):
        eq = pd.Series(np.exp(np.cumsum(rng.normal(0, 0.02, 300))))
        assert -1.0 < max_drawdown(eq) <= 0.0


def test_total_return_and_cagr():
    eq = pd.Series(np.linspace(1.0, 2.0, 252))
    assert total_return(eq) == pytest.approx(1.0)
    # Doubling over exactly one year of bars is a 100% CAGR.
    assert cagr(eq, periods_per_year=252) == pytest.approx(1.0)
    # The same doubling over two years is sqrt(2) - 1.
    eq2 = pd.Series(np.linspace(1.0, 2.0, 504))
    assert cagr(eq2) == pytest.approx(math.sqrt(2) - 1)


def test_sharpe_matches_manual_formula():
    r = pd.Series([0.01, -0.005, 0.002, 0.007, -0.003])
    expected = math.sqrt(252) * r.mean() / r.std(ddof=1)
    assert sharpe_ratio(r) == pytest.approx(expected)
    assert sharpe_ratio(r, periods_per_year=12) == pytest.approx(math.sqrt(12) * r.mean() / r.std())


def test_sharpe_undefined_when_no_variance():
    # A constant return series has no risk estimate; returning 0 would be misleading.
    assert math.isnan(sharpe_ratio(pd.Series([0.01] * 10)))
    assert math.isnan(sharpe_ratio(pd.Series([0.0] * 10)))


def test_sharpe_undefined_for_single_observation():
    assert math.isnan(sharpe_ratio(pd.Series([0.01])))


def test_sharpe_sign_and_scale_invariance():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.001, 0.01, 500))
    assert sharpe_ratio(-r) == pytest.approx(-sharpe_ratio(r))
    assert sharpe_ratio(3 * r) == pytest.approx(sharpe_ratio(r))


def test_annualised_volatility():
    r = pd.Series([0.01, -0.01] * 50)
    assert annualised_volatility(r) == pytest.approx(r.std(ddof=1) * math.sqrt(252))


def test_sharpe_standard_error_shrinks_with_more_data():
    se_short = sharpe_standard_error(1.0, 252)
    se_long = sharpe_standard_error(1.0, 252 * 16)
    assert se_long == pytest.approx(se_short / 4, rel=1e-3)
    # One year of daily data: SE of an annualised Sharpe is roughly 1.
    assert sharpe_standard_error(0.0, 252) == pytest.approx(1.0)
    assert math.isnan(sharpe_standard_error(float("nan"), 252))


def test_sharpe_standard_error_is_calibrated_on_iid_normal_data():
    rng = np.random.default_rng(2)
    estimates = [sharpe_ratio(pd.Series(rng.normal(0, 0.01, 252))) for _ in range(2000)]
    assert np.std(estimates) == pytest.approx(sharpe_standard_error(0.0, 252), rel=0.08)


def test_summarise_returns_keys_and_consistency():
    r = pd.Series([0.01, -0.02, 0.015, 0.0, 0.005])
    s = summarise_returns(r)
    assert set(s) == {"total_return", "cagr", "ann_vol", "sharpe", "sharpe_se", "max_drawdown"}
    assert s["total_return"] == pytest.approx(float((1 + r).prod() - 1))
    assert s["sharpe"] == pytest.approx(sharpe_ratio(r))


def test_expected_max_sharpe_grows_with_trials():
    one = expected_max_sharpe_under_null(1, 756)
    ten = expected_max_sharpe_under_null(10, 756)
    hundred = expected_max_sharpe_under_null(100, 756)
    assert one == 0.0
    assert 0 < ten < hundred


def test_expected_max_sharpe_matches_simulation():
    rng = np.random.default_rng(3)
    n_obs, k = 504, 50
    best = [
        max(sharpe_ratio(pd.Series(rng.normal(0, 0.01, n_obs))) for _ in range(k))
        for _ in range(200)
    ]
    assert np.mean(best) == pytest.approx(expected_max_sharpe_under_null(k, n_obs), rel=0.1)


@pytest.mark.parametrize(
    "func, arg",
    [
        (max_drawdown, pd.Series([], dtype=float)),
        (max_drawdown, pd.Series([1.0, float("nan"), 1.1])),
        (max_drawdown, pd.Series([1.0, 0.0, 1.1])),
        (max_drawdown, [1.0, 1.1]),
        (sharpe_ratio, pd.Series([0.01, float("inf")])),
        (sharpe_ratio, pd.Series(["a", "b"])),
        (equity_curve, pd.Series([0.1, -1.0])),
        (simple_returns, pd.Series([100.0, -5.0])),
    ],
)
def test_invalid_inputs_raise_clear_errors(func, arg):
    with pytest.raises(BacktestInputError):
        func(arg)


def test_error_message_names_the_problem():
    with pytest.raises(BacktestInputError, match="NaN/inf"):
        sharpe_ratio(pd.Series([0.01, float("nan"), 0.02]))
    with pytest.raises(BacktestInputError, match="periods_per_year"):
        sharpe_ratio(pd.Series([0.01, 0.02]), periods_per_year=0)
