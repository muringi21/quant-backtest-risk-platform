"""Performance metrics for a series of per-period simple returns.

Conventions (see ``docs/methodology.md`` for the reasoning):

* Returns are simple (not log) returns per bar; equity compounds them.
* Annualisation uses ``periods_per_year`` (252 trading days by default) and
  assumes returns are independent and identically distributed, which real
  returns are not. Treat annualised figures as conventions, not facts.
* The risk-free rate is taken as zero, so "Sharpe" here is mean / std of the
  strategy's returns.
* Undefined quantities (e.g. Sharpe of a constant series) return ``nan``
  rather than a misleading number such as 0.
"""

from __future__ import annotations

import math

import pandas as pd

from .errors import BacktestInputError
from .validation import require_float, require_int, validate_prices, validate_returns

TRADING_DAYS = 252


def simple_returns(prices: pd.Series) -> pd.Series:
    """Per-bar simple returns ``p[t] / p[t-1] - 1``; the first bar is defined as 0."""
    p = validate_prices(prices, "prices")
    return p.pct_change().fillna(0.0).rename("asset_ret")


# Backwards-compatible name from v0.1.
daily_returns = simple_returns


def equity_curve(strategy_returns: pd.Series, start: float = 1.0) -> pd.Series:
    """Compound per-bar returns into an equity curve starting from ``start``."""
    r = validate_returns(strategy_returns, "strategy_returns")
    start = require_float(start, "start", minimum=0.0, strict_minimum=True)
    return (start * (1.0 + r).cumprod()).rename("equity")


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Fractional distance below the running peak at each bar (0 at new highs)."""
    eq = validate_prices(equity, "equity", min_length=1)
    return (eq / eq.cummax() - 1.0).rename("drawdown")


def max_drawdown(equity: pd.Series) -> float:
    """Worst peak-to-trough decline as a non-positive fraction, e.g. -0.25 for -25%."""
    return float(drawdown_series(equity).min())


def total_return(equity: pd.Series, start: float = 1.0) -> float:
    eq = validate_prices(equity, "equity", min_length=1)
    return float(eq.iloc[-1] / start - 1.0)


def cagr(equity: pd.Series, periods_per_year: int = TRADING_DAYS, start: float = 1.0) -> float:
    """Compound annual growth rate implied by ``equity``.

    ``equity`` is assumed to be produced from one return per bar with the
    starting capital ``start`` held *before* the first bar, so the number of
    compounding periods equals ``len(equity)``.
    """
    eq = validate_prices(equity, "equity", min_length=1)
    ppy = require_int(periods_per_year, "periods_per_year", minimum=1)
    years = len(eq) / ppy
    return float((eq.iloc[-1] / start) ** (1.0 / years) - 1.0)


def annualised_volatility(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    r = validate_returns(returns, "returns", min_length=2)
    ppy = require_int(periods_per_year, "periods_per_year", minimum=1)
    return float(r.std(ddof=1) * math.sqrt(ppy))


def sharpe_ratio(strategy_returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """Annualised Sharpe ratio with a zero risk-free rate.

    Returns ``nan`` when the ratio is undefined: fewer than two observations
    or (numerically) zero variance, e.g. a strategy that never trades.
    """
    r = validate_returns(strategy_returns, "strategy_returns")
    ppy = require_int(periods_per_year, "periods_per_year", minimum=1)
    if len(r) < 2:
        return float("nan")
    std = float(r.std(ddof=1))
    if std < 1e-12:
        return float("nan")
    return float(math.sqrt(ppy) * float(r.mean()) / std)


def sharpe_standard_error(
    sharpe: float, n_obs: int, periods_per_year: int = TRADING_DAYS
) -> float:
    """Approximate standard error of an *annualised* Sharpe ratio.

    Uses the i.i.d. approximation of Lo (2002): for a per-period Sharpe ``s``
    estimated from ``T`` observations, ``SE(s) ~= sqrt((1 + s**2 / 2) / T)``,
    then scales by ``sqrt(periods_per_year)``. Fat tails, skew and serial
    correlation all make the true uncertainty larger than this.
    """
    n = require_int(n_obs, "n_obs", minimum=2)
    ppy = require_int(periods_per_year, "periods_per_year", minimum=1)
    if not math.isfinite(sharpe):
        return float("nan")
    per_period = sharpe / math.sqrt(ppy)
    return float(math.sqrt((1.0 + 0.5 * per_period**2) / n) * math.sqrt(ppy))


def summarise_returns(
    returns: pd.Series, periods_per_year: int = TRADING_DAYS
) -> dict[str, float]:
    """Headline metrics for a return series (used for in/out-of-sample slices)."""
    r = validate_returns(returns, "returns", min_length=2)
    eq = equity_curve(r)
    sr = sharpe_ratio(r, periods_per_year)
    return {
        "total_return": total_return(eq),
        "cagr": cagr(eq, periods_per_year),
        "ann_vol": annualised_volatility(r, periods_per_year),
        "sharpe": sr,
        "sharpe_se": sharpe_standard_error(sr, len(r), periods_per_year),
        "max_drawdown": max_drawdown(eq),
    }


def expected_max_sharpe_under_null(
    n_trials: int, n_obs: int, periods_per_year: int = TRADING_DAYS
) -> float:
    """Expected best annualised Sharpe from ``n_trials`` *independent* zero-edge strategies.

    Uses the extreme-value approximation from Bailey & Lopez de Prado (2014),
    with each trial's Sharpe ~ N(0, 1/T) per period. Real parameter sweeps
    produce correlated trials, so treat this as an order-of-magnitude guide.
    """
    k = require_int(n_trials, "n_trials", minimum=1)
    n = require_int(n_obs, "n_obs", minimum=2)
    ppy = require_int(periods_per_year, "periods_per_year", minimum=1)
    if k == 1:
        return 0.0
    from statistics import NormalDist

    gamma = 0.5772156649015329  # Euler-Mascheroni
    z = NormalDist()
    e_max_z = (1 - gamma) * z.inv_cdf(1 - 1 / k) + gamma * z.inv_cdf(1 - 1 / (k * math.e))
    return float(e_max_z * math.sqrt(ppy / n))


__all__ = [
    "TRADING_DAYS",
    "BacktestInputError",
    "annualised_volatility",
    "cagr",
    "daily_returns",
    "drawdown_series",
    "equity_curve",
    "expected_max_sharpe_under_null",
    "max_drawdown",
    "sharpe_ratio",
    "sharpe_standard_error",
    "simple_returns",
    "summarise_returns",
    "total_return",
]

