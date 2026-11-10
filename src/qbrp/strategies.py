"""Signal functions that map a price series to a target exposure in [-1, 1].

Every strategy here uses only prices up to and including bar ``t`` to decide
``position[t]``; the engine is responsible for delaying execution. During a
warm-up period, when an indicator is not yet defined, the position is 0 (flat).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .errors import BacktestInputError
from .validation import require_int, validate_prices


@dataclass(frozen=True)
class MACrossConfig:
    """Moving-average crossover parameters.

    ``fast`` and ``slow`` are window lengths in bars, with ``fast < slow``.
    If ``long_only`` is true the strategy is flat instead of short when the
    fast average is below the slow one.
    """

    fast: int = 20
    slow: int = 50
    long_only: bool = False

    def __post_init__(self) -> None:
        require_int(self.fast, "fast", minimum=1)
        require_int(self.slow, "slow", minimum=2)
        if self.fast >= self.slow:
            raise BacktestInputError(
                f"fast window ({self.fast}) must be shorter than slow window ({self.slow})"
            )


@dataclass(frozen=True)
class MomentumConfig:
    """Time-series momentum: hold the sign of the trailing ``lookback``-bar return."""

    lookback: int = 126
    long_only: bool = False

    def __post_init__(self) -> None:
        require_int(self.lookback, "lookback", minimum=1)


def _apply_long_only(signal: pd.Series, long_only: bool) -> pd.Series:
    return signal.clip(lower=0.0) if long_only else signal


def ma_crossover(prices: pd.Series, cfg: MACrossConfig) -> pd.Series:
    """+1 when the fast SMA is above the slow SMA, -1 when below, 0 when equal or warming up."""
    p = validate_prices(prices, "prices")
    if len(p) < cfg.slow:
        raise BacktestInputError(
            f"need at least slow={cfg.slow} prices for the moving averages, got {len(p)}"
        )
    fast_ma = p.rolling(cfg.fast).mean()
    slow_ma = p.rolling(cfg.slow).mean()
    signal = np.sign(fast_ma - slow_ma).fillna(0.0)
    return _apply_long_only(signal, cfg.long_only).astype(float).rename("position")


def momentum(prices: pd.Series, cfg: MomentumConfig) -> pd.Series:
    """Sign of the return over the previous ``lookback`` bars (0 during warm-up)."""
    p = validate_prices(prices, "prices")
    if len(p) <= cfg.lookback:
        raise BacktestInputError(
            f"need more than lookback={cfg.lookback} prices, got {len(p)}"
        )
    signal = np.sign(p / p.shift(cfg.lookback) - 1.0).fillna(0.0)
    return _apply_long_only(signal, cfg.long_only).astype(float).rename("position")


def buy_and_hold(prices: pd.Series) -> pd.Series:
    """Fully long from the first bar: the benchmark every timing rule should beat."""
    p = validate_prices(prices, "prices")
    return pd.Series(1.0, index=p.index, name="position")
