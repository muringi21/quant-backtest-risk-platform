"""Input checks shared by the data loaders, strategies, engine and metrics.

Every check raises :class:`~qbrp.errors.BacktestInputError` with a message that
names the offending argument, so a bad input fails loudly instead of quietly
producing a plausible-looking number.
"""

from __future__ import annotations

import math
from numbers import Integral, Real

import numpy as np
import pandas as pd

from .errors import BacktestInputError


def require_series(obj: object, name: str) -> pd.Series:
    if not isinstance(obj, pd.Series):
        raise BacktestInputError(f"{name} must be a pandas Series, got {type(obj).__name__}")
    return obj


def require_int(value: object, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise BacktestInputError(f"{name} must be an integer, got {value!r}")
    value = int(value)
    if minimum is not None and value < minimum:
        raise BacktestInputError(f"{name} must be >= {minimum}, got {value}")
    return value


def require_float(
    value: object,
    name: str,
    *,
    minimum: float | None = None,
    strict_minimum: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise BacktestInputError(f"{name} must be a number, got {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise BacktestInputError(f"{name} must be finite, got {value}")
    if minimum is not None:
        if strict_minimum and value <= minimum:
            raise BacktestInputError(f"{name} must be > {minimum}, got {value}")
        if not strict_minimum and value < minimum:
            raise BacktestInputError(f"{name} must be >= {minimum}, got {value}")
    return value


def _check_numeric_finite(s: pd.Series, name: str) -> None:
    if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        raise BacktestInputError(f"{name} must be numeric, got dtype {s.dtype}")
    values = s.to_numpy(dtype=float)
    bad = ~np.isfinite(values)
    if bad.any():
        first = s.index[int(np.argmax(bad))]
        raise BacktestInputError(
            f"{name} contains {int(bad.sum())} NaN/inf value(s); first at index {first!r}"
        )


def validate_prices(prices: object, name: str = "prices", *, min_length: int = 2) -> pd.Series:
    """Check a price series is usable for a backtest and return it as float.

    Requirements: a numeric Series of at least ``min_length`` finite, strictly
    positive values with a unique, strictly increasing index.
    """
    s = require_series(prices, name)
    if len(s) < min_length:
        raise BacktestInputError(f"{name} needs at least {min_length} observations, got {len(s)}")
    _check_numeric_finite(s, name)
    if (s <= 0).any():
        first = s.index[int(np.argmax((s <= 0).to_numpy()))]
        raise BacktestInputError(f"{name} must be strictly positive; first bad value at {first!r}")
    if not s.index.is_unique:
        dup = s.index[s.index.duplicated()][0]
        raise BacktestInputError(f"{name} index has duplicate labels, e.g. {dup!r}")
    if not s.index.is_monotonic_increasing:
        raise BacktestInputError(f"{name} index must be sorted in increasing (time) order")
    return s.astype(float)


def validate_returns(returns: object, name: str = "returns", *, min_length: int = 1) -> pd.Series:
    """Check a return series is numeric, finite and not shorter than ``min_length``."""
    s = require_series(returns, name)
    if len(s) < min_length:
        raise BacktestInputError(f"{name} needs at least {min_length} observations, got {len(s)}")
    _check_numeric_finite(s, name)
    if (s <= -1.0).any():
        raise BacktestInputError(f"{name} contains a return <= -100%, which wipes out equity")
    return s.astype(float)


def validate_positions(position: object, prices: pd.Series, name: str = "position") -> pd.Series:
    """Check target positions line up with ``prices`` and lie in [-1, 1]."""
    s = require_series(position, name)
    if not s.index.equals(prices.index):
        raise BacktestInputError(
            f"{name} index must match the price index exactly "
            f"(got {len(s)} labels vs {len(prices)}); reindex explicitly before calling"
        )
    _check_numeric_finite(s, name)
    if (s.abs() > 1.0 + 1e-12).any():
        raise BacktestInputError(
            f"{name} values must lie in [-1, 1] (leverage is not modelled); "
            f"max |position| = {float(s.abs().max()):g}"
        )
    return s.astype(float)
