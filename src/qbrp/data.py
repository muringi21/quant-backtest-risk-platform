"""Price inputs: a seeded synthetic generator and a validated CSV loader."""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .errors import BacktestInputError
from .validation import require_float, require_int, validate_prices

DEFAULT_START = "2020-01-01"


def simulate_log_returns(
    n: int,
    seed: int,
    drift: float = 0.0003,
    vol: float = 0.01,
    trend_share: float = 0.0,
    trend_persistence: float = 0.98,
) -> np.ndarray:
    """Daily log returns: constant drift + a slowly varying hidden trend + noise.

    ``r[t] = drift + x[t] + e[t]`` where ``x`` is a stationary AR(1) "trend"
    component, ``x[t] = trend_persistence * x[t-1] + u[t]``, and ``e`` is i.i.d.
    Gaussian noise. ``trend_share`` is the fraction of return variance that
    comes from ``x``; the total standard deviation of ``r`` is always ``vol``.

    * ``trend_share=0`` (default) is a geometric random walk with drift: past
      prices carry no information about future returns, so no timing rule has
      a genuine edge over buy-and-hold.
    * ``trend_share>0`` plants a weak, known trend. Return autocorrelation at
      lag ``k`` is ``trend_share * trend_persistence**k``, e.g. only ~0.02 at
      lag 1 for the experiment defaults, yet it accumulates over weeks, which
      is what trend-following rules exploit.
    """
    n = require_int(n, "n", minimum=2)
    seed = require_int(seed, "seed", minimum=0)
    drift = require_float(drift, "drift")
    vol = require_float(vol, "vol", minimum=0.0)
    trend_share = require_float(trend_share, "trend_share", minimum=0.0)
    trend_persistence = require_float(trend_persistence, "trend_persistence", minimum=0.0)
    if trend_share >= 1.0:
        raise BacktestInputError(f"trend_share must be in [0, 1), got {trend_share}")
    if trend_persistence >= 1.0:
        raise BacktestInputError(
            f"trend_persistence must be in [0, 1), got {trend_persistence}"
        )

    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n) * vol * math.sqrt(1.0 - trend_share)
    if trend_share == 0.0:
        return drift + noise
    trend_sd = vol * math.sqrt(trend_share)
    innovations = rng.standard_normal(n) * trend_sd * math.sqrt(1.0 - trend_persistence**2)
    trend = np.empty(n)
    # Start from the stationary distribution so there is no burn-in artefact.
    prev = rng.standard_normal() * trend_sd
    for t in range(n):
        prev = trend_persistence * prev + innovations[t]
        trend[t] = prev
    return drift + trend + noise


def make_sample_prices(
    n: int = 756,
    seed: int = 7,
    drift: float = 0.0003,
    vol: float = 0.01,
    trend_share: float = 0.0,
    trend_persistence: float = 0.98,
    start: str = DEFAULT_START,
    start_price: float = 100.0,
) -> pd.DataFrame:
    """Synthetic daily close prices on a business-day calendar.

    The calendar starts at a fixed date (not "today") so that the same
    arguments always produce exactly the same DataFrame, index included.
    Defaults (0.03% mean, 1% std daily log return) roughly mimic a broad
    equity index: ~7.5% drift and ~16% volatility a year.
    """
    start_price = require_float(start_price, "start_price", minimum=0.0, strict_minimum=True)
    rets = simulate_log_returns(
        n,
        seed,
        drift=drift,
        vol=vol,
        trend_share=trend_share,
        trend_persistence=trend_persistence,
    )
    price = start_price * np.exp(np.cumsum(rets))
    try:
        idx = pd.bdate_range(start=pd.Timestamp(start), periods=n, name="date")
    except (ValueError, TypeError) as exc:
        raise BacktestInputError(f"start must be a date such as '2020-01-01', got {start!r}") from exc
    return pd.DataFrame({"close": price}, index=idx)


def load_prices_csv(
    path: str | Path,
    column: str = "close",
    date_column: str | None = "date",
) -> pd.DataFrame:
    """Load a price column from a CSV file and validate it.

    The file must contain ``column``. If ``date_column`` is present it is parsed
    as dates and used as the index; rows are sorted by it. Pass
    ``date_column=None`` to use row order. Columns are matched case-insensitively.
    Bad data (missing values, non-numeric or non-positive prices, duplicate
    dates) raises :class:`BacktestInputError` rather than being dropped.
    """
    path = Path(path)
    if not path.is_file():
        raise BacktestInputError(f"CSV file not found: {path}")
    try:
        df = pd.read_csv(path)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise BacktestInputError(f"could not parse {path} as CSV: {exc}") from exc

    lower = {str(c).lower(): c for c in df.columns}
    if column.lower() not in lower:
        raise BacktestInputError(
            f"column {column!r} not found in {path.name}; available: {list(df.columns)}"
        )
    close = df[lower[column.lower()]]

    if date_column is not None and date_column.lower() in lower:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                dates = pd.to_datetime(df[lower[date_column.lower()]], errors="raise")
        except (ValueError, TypeError) as exc:
            raise BacktestInputError(f"could not parse {date_column!r} as dates: {exc}") from exc
        close = pd.Series(close.to_numpy(), index=pd.DatetimeIndex(dates, name="date"))
        close = close.sort_index()

    if not pd.api.types.is_numeric_dtype(close):
        numeric = pd.to_numeric(close, errors="coerce")
        bad = numeric.isna() & close.notna()
        if bad.any():
            first = close[bad].iloc[0]
            raise BacktestInputError(
                f"column {column!r} has {int(bad.sum())} non-numeric value(s), e.g. {first!r}"
            )
        close = numeric
    close = validate_prices(close.rename("close"), f"{path.name}:{column}")
    return close.to_frame("close")
