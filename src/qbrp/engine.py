"""Vectorised single-asset backtest engine.

Timing model
------------
All quantities are indexed by bar ``t`` (one row of the price series, e.g.
one trading day). ``r[t] = p[t] / p[t-1] - 1`` is the asset's simple return.

* ``position[t]`` is the *target* exposure in [-1, 1] that a strategy decides
  using information available at the close of bar ``t`` (including ``p[t]``).
* ``execution_lag`` is the number of bars between that decision and the trade.
  The exposure actually held at the end of bar ``t`` is

      held[t] = position[t - execution_lag]      (0 before the first signal)

  ``execution_lag=0`` trades at the same close used to compute the signal,
  which is optimistic (it assumes you can observe the close and trade at it).
  ``execution_lag=1`` (the default) trades at the next bar's close.
* The strategy earns the return on the exposure held *into* the bar, minus
  costs for any trade made at that bar's close:

      gross[t] = held[t-1] * r[t]
      cost[t]  = fee_bps / 10_000 * |held[t] - held[t-1]|
      net[t]   = gross[t] - cost[t]

Because ``held[t-1]`` depends only on prices up to ``t-1-execution_lag``, a
strategy whose ``position[t]`` uses only data up to ``t`` cannot see the return
it is paid on. ``tests/test_engine.py`` checks this property directly.

Simplifications (documented in ``docs/methodology.md``): fills at the close
price, a fixed proportional cost per unit of turnover, no slippage model
beyond that cost, free and unlimited shorting, no financing or borrow costs,
no leverage, and daily rebalancing to the target exposure.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .errors import BacktestInputError
from .metrics import (
    TRADING_DAYS,
    annualised_volatility,
    cagr,
    drawdown_series,
    equity_curve,
    max_drawdown,
    sharpe_ratio,
    sharpe_standard_error,
    simple_returns,
)
from .validation import require_float, require_int, validate_positions, validate_prices


@dataclass
class BacktestResult:
    """Output of :func:`run_backtest`.

    The per-bar series share the price index. Scalar metrics are computed
    from ``strategy_returns`` / ``equity`` and follow the conventions in
    :mod:`qbrp.metrics`.
    """

    equity: pd.Series
    strategy_returns: pd.Series
    total_return: float
    max_drawdown: float
    sharpe: float
    held_position: pd.Series = field(repr=False)
    gross_returns: pd.Series = field(repr=False)
    costs: pd.Series = field(repr=False)
    turnover: pd.Series = field(repr=False)
    drawdown: pd.Series = field(repr=False)
    cagr: float = float("nan")
    ann_vol: float = float("nan")
    sharpe_se: float = float("nan")
    n_trades: int = 0
    total_turnover: float = 0.0
    annual_turnover: float = 0.0
    exposure: float = 0.0
    total_cost: float = 0.0
    fee_bps: float = 0.0
    execution_lag: int = 1
    periods_per_year: int = TRADING_DAYS

    def summary(self) -> dict[str, float | int]:
        """Scalar results as a plain dict (JSON-serialisable)."""
        return {
            "total_return": self.total_return,
            "cagr": self.cagr,
            "ann_vol": self.ann_vol,
            "sharpe": self.sharpe,
            "sharpe_se": self.sharpe_se,
            "max_drawdown": self.max_drawdown,
            "n_trades": self.n_trades,
            "total_turnover": self.total_turnover,
            "annual_turnover": self.annual_turnover,
            "exposure": self.exposure,
            "total_cost": self.total_cost,
            "fee_bps": self.fee_bps,
            "execution_lag": self.execution_lag,
            "n_bars": len(self.equity),
        }


def held_exposure(position: pd.Series, execution_lag: int) -> pd.Series:
    """Exposure held at the end of each bar after delaying ``position`` by ``execution_lag``."""
    return position.shift(execution_lag).fillna(0.0).rename("held_position")


def run_backtest(
    close: pd.Series,
    position: pd.Series,
    fee_bps: float = 1.0,
    execution_lag: int = 1,
    periods_per_year: int = TRADING_DAYS,
) -> BacktestResult:
    """Backtest a target-exposure series against a single price series.

    Parameters
    ----------
    close:
        Strictly positive prices with a sorted, unique index.
    position:
        Target exposure per bar in [-1, 1] (1 = fully long, -1 = fully short),
        on exactly the same index as ``close``. No NaNs are allowed: decide
        explicitly what the strategy holds during any warm-up period.
    fee_bps:
        Cost in basis points charged per unit of turnover. Moving from +1 to
        -1 is two units of turnover and costs ``2 * fee_bps``.
    execution_lag:
        Bars between a signal and its execution (see module docstring).
    periods_per_year:
        Used only to annualise metrics.
    """
    close = validate_prices(close, "close")
    position = validate_positions(position, close, "position")
    fee_bps = require_float(fee_bps, "fee_bps", minimum=0.0)
    execution_lag = require_int(execution_lag, "execution_lag", minimum=0)
    periods_per_year = require_int(periods_per_year, "periods_per_year", minimum=1)
    if execution_lag >= len(close):
        raise BacktestInputError(
            f"execution_lag ({execution_lag}) must be smaller than the number of bars "
            f"({len(close)})"
        )

    asset_ret = simple_returns(close)
    held = held_exposure(position, execution_lag)
    held_prev = held.shift(1).fillna(0.0)

    turnover = (held - held_prev).abs().rename("turnover")
    costs = (fee_bps / 10_000.0 * turnover).rename("cost")
    gross = (held_prev * asset_ret).rename("gross_ret")
    net = (gross - costs).rename("strategy_ret")

    eq = equity_curve(net, start=1.0)
    sr = sharpe_ratio(net, periods_per_year)
    years = len(close) / periods_per_year

    return BacktestResult(
        equity=eq,
        strategy_returns=net,
        total_return=float(eq.iloc[-1] - 1.0),
        max_drawdown=max_drawdown(eq),
        sharpe=sr,
        held_position=held,
        gross_returns=gross,
        costs=costs,
        turnover=turnover,
        drawdown=drawdown_series(eq),
        cagr=cagr(eq, periods_per_year),
        ann_vol=annualised_volatility(net, periods_per_year),
        sharpe_se=sharpe_standard_error(sr, len(net), periods_per_year),
        n_trades=int((turnover > 1e-12).sum()),
        total_turnover=float(turnover.sum()),
        annual_turnover=float(turnover.sum() / years),
        exposure=float(held.abs().mean()),
        total_cost=float(costs.sum()),
        fee_bps=fee_bps,
        execution_lag=execution_lag,
        periods_per_year=periods_per_year,
    )
