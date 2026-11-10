"""Reproducible synthetic experiments.

Three studies, each a pure function of its arguments (seeds are explicit and
every strategy sees the same simulated paths, i.e. common random numbers):

``compare_strategies``
    Buy-and-hold vs. timing rules on many simulated price paths in two
    worlds: a random walk with drift (no rule has an edge) and the same walk
    with a weak hidden trend (trend rules have a small, real edge).
``fee_lag_sensitivity``
    How the edge in the trending world decays as costs and execution delay rise.
``overfitting_study``
    The naive-backtest trap: choose the best moving-average pair from a grid
    on the first half of a path with *no* edge, then look at the second half.

Run them with ``qbrp experiment all`` (see the README); outputs are CSV files,
a Markdown report and, if matplotlib is installed, figures.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .data import make_sample_prices
from .engine import run_backtest
from .metrics import TRADING_DAYS, expected_max_sharpe_under_null, sharpe_ratio
from .strategies import MACrossConfig, MomentumConfig, buy_and_hold, ma_crossover, momentum
from .validation import require_float, require_int


@dataclass(frozen=True)
class Regime:
    name: str
    description: str
    drift: float = 0.0003
    vol: float = 0.01
    trend_share: float = 0.0
    trend_persistence: float = 0.98

    def prices(self, n_bars: int, seed: int) -> pd.Series:
        return make_sample_prices(
            n=n_bars,
            seed=seed,
            drift=self.drift,
            vol=self.vol,
            trend_share=self.trend_share,
            trend_persistence=self.trend_persistence,
        )["close"]


REGIMES: dict[str, Regime] = {
    "random_walk": Regime(
        "random_walk",
        "Geometric random walk, ~7.5%/yr drift, ~16%/yr vol. Past prices carry no "
        "information about future returns.",
    ),
    "weak_trend": Regime(
        "weak_trend",
        "Same drift and vol, but 2% of daily return variance comes from a hidden "
        "trend with persistence 0.98 (lag-1 autocorrelation ~0.02).",
        trend_share=0.02,
    ),
    "no_edge": Regime(
        "no_edge",
        "Zero-drift random walk: every rule, long or short, has an expected return "
        "of zero before costs.",
        drift=0.0,
    ),
}

StrategyFn = Callable[[pd.Series], pd.Series]

STRATEGIES: dict[str, tuple[str, StrategyFn]] = {
    "buy_and_hold": ("Buy and hold", buy_and_hold),
    "ma_5_20": ("MA 5/20 long-short", lambda p: ma_crossover(p, MACrossConfig(5, 20))),
    "ma_20_50": ("MA 20/50 long-short", lambda p: ma_crossover(p, MACrossConfig(20, 50))),
    "ma_20_50_long": (
        "MA 20/50 long-only",
        lambda p: ma_crossover(p, MACrossConfig(20, 50, long_only=True)),
    ),
    "momentum_126": ("Momentum 126d long-short", lambda p: momentum(p, MomentumConfig(126))),
}


def _seeds(n_seeds: int, base_seed: int) -> range:
    n_seeds = require_int(n_seeds, "n_seeds", minimum=1)
    base_seed = require_int(base_seed, "base_seed", minimum=0)
    return range(base_seed, base_seed + n_seeds)


# --------------------------------------------------------------------------- #
# 1. Strategy comparison
# --------------------------------------------------------------------------- #
def compare_strategies(
    n_seeds: int = 200,
    n_bars: int = 2520,
    fee_bps: float = 1.0,
    execution_lag: int = 1,
    regimes: Sequence[str] = ("random_walk", "weak_trend"),
    strategies: Sequence[str] = tuple(STRATEGIES),
    base_seed: int = 0,
) -> pd.DataFrame:
    """One row per (regime, strategy, seed) with that run's summary metrics."""
    rows = []
    for regime_name in regimes:
        regime = REGIMES[regime_name]
        for seed in _seeds(n_seeds, base_seed):
            close = regime.prices(n_bars, seed)
            for key in strategies:
                _, fn = STRATEGIES[key]
                res = run_backtest(close, fn(close), fee_bps=fee_bps, execution_lag=execution_lag)
                rows.append({"regime": regime_name, "strategy": key, "seed": seed, **res.summary()})
    return pd.DataFrame(rows)


def summarise_comparison(runs: pd.DataFrame) -> pd.DataFrame:
    """Distribution of results across paths, plus a paired comparison with buy-and-hold."""
    bh = runs[runs["strategy"] == "buy_and_hold"].set_index(["regime", "seed"])["sharpe"]
    runs = runs.join(bh.rename("bh_sharpe"), on=["regime", "seed"])
    runs["beats_bh"] = runs["sharpe"] > runs["bh_sharpe"]
    g = runs.groupby(["regime", "strategy"], sort=False)
    out = pd.DataFrame(
        {
            "mean_sharpe": g["sharpe"].mean(),
            "sharpe_p05": g["sharpe"].quantile(0.05),
            "sharpe_p95": g["sharpe"].quantile(0.95),
            "mean_cagr": g["cagr"].mean(),
            "mean_max_drawdown": g["max_drawdown"].mean(),
            "annual_turnover": g["annual_turnover"].mean(),
            "pct_sharpe_gt_0": g["sharpe"].apply(lambda s: float((s > 0).mean())),
            "pct_beats_bh": g["beats_bh"].mean(),
        }
    ).reset_index()
    out.loc[out["strategy"] == "buy_and_hold", "pct_beats_bh"] = np.nan
    return out


# --------------------------------------------------------------------------- #
# 2. Fee and execution-lag sensitivity
# --------------------------------------------------------------------------- #
def fee_lag_sensitivity(
    n_seeds: int = 200,
    n_bars: int = 2520,
    regime: str = "weak_trend",
    strategies: Sequence[str] = ("ma_5_20", "ma_20_50"),
    fees_bps: Sequence[float] = (0.0, 2.0, 5.0, 10.0, 20.0, 50.0),
    lags: Sequence[int] = (0, 1, 2, 5, 10, 20),
    base_seed: int = 0,
) -> pd.DataFrame:
    """Mean Sharpe (and CAGR) over paths for every (strategy, fee, lag) combination."""
    reg = REGIMES[regime]
    rows = []
    for seed in _seeds(n_seeds, base_seed):
        close = reg.prices(n_bars, seed)
        bh = run_backtest(close, buy_and_hold(close), fee_bps=0.0, execution_lag=0)
        for key in strategies:
            pos = STRATEGIES[key][1](close)
            for fee in fees_bps:
                for lag in lags:
                    res = run_backtest(close, pos, fee_bps=fee, execution_lag=lag)
                    rows.append(
                        {
                            "strategy": key,
                            "fee_bps": float(fee),
                            "execution_lag": int(lag),
                            "seed": seed,
                            "sharpe": res.sharpe,
                            "cagr": res.cagr,
                            "annual_turnover": res.annual_turnover,
                            "bh_sharpe": bh.sharpe,
                        }
                    )
    df = pd.DataFrame(rows)
    return (
        df.groupby(["strategy", "fee_bps", "execution_lag"])
        .agg(
            mean_sharpe=("sharpe", "mean"),
            mean_cagr=("cagr", "mean"),
            annual_turnover=("annual_turnover", "mean"),
            bh_mean_sharpe=("bh_sharpe", "mean"),
        )
        .reset_index()
    )


# --------------------------------------------------------------------------- #
# 3. Overfitting a parameter grid
# --------------------------------------------------------------------------- #
DEFAULT_FAST = tuple(range(5, 55, 5))
DEFAULT_SLOW = tuple(range(20, 210, 10))


def ma_grid(fast: Sequence[int] = DEFAULT_FAST, slow: Sequence[int] = DEFAULT_SLOW):
    return [(f, s) for f in fast for s in slow if f < s]


def _rolling_means(close: np.ndarray, windows: set[int]) -> dict[int, np.ndarray]:
    csum = np.concatenate([[0.0], np.cumsum(close)])
    out = {}
    for w in windows:
        m = np.full(close.shape, np.nan)
        m[w - 1 :] = (csum[w:] - csum[:-w]) / w
        out[w] = m
    return out


def fast_net_returns(
    asset_ret: np.ndarray, position: np.ndarray, fee_bps: float, execution_lag: int
) -> np.ndarray:
    """NumPy re-implementation of the engine's return formula for large sweeps.

    Skips validation; ``tests/test_experiments.py`` checks it matches
    :func:`qbrp.engine.run_backtest` exactly.
    """
    held = np.zeros_like(position)
    if execution_lag == 0:
        held[:] = position
    else:
        held[execution_lag:] = position[:-execution_lag]
    prev = np.concatenate([[0.0], held[:-1]])
    return prev * asset_ret - fee_bps / 10_000.0 * np.abs(held - prev)


def _sharpe_np(r: np.ndarray, ppy: int = TRADING_DAYS) -> float:
    sd = r.std(ddof=1)
    return float(np.sqrt(ppy) * r.mean() / sd) if sd > 1e-12 else float("nan")


def grid_in_out_sample(
    close: pd.Series,
    grid: Sequence[tuple[int, int]],
    split: float = 0.5,
    fee_bps: float = 1.0,
    execution_lag: int = 1,
) -> pd.DataFrame:
    """In- and out-of-sample Sharpe for every MA pair on one path.

    Signals are computed on the full series (they only use past prices) and
    the resulting strategy returns are split in time, so the out-of-sample
    half does not lose bars to indicator warm-up.
    """
    split = require_float(split, "split", minimum=0.0, strict_minimum=True)
    p = close.to_numpy(dtype=float)
    r = np.concatenate([[0.0], p[1:] / p[:-1] - 1.0])
    cut = int(len(p) * split)
    mas = _rolling_means(p, {w for pair in grid for w in pair})
    rows = []
    for f, s in grid:
        pos = np.nan_to_num(np.sign(mas[f] - mas[s]))
        net = fast_net_returns(r, pos, fee_bps, execution_lag)
        rows.append(
            {
                "fast": f,
                "slow": s,
                "is_sharpe": _sharpe_np(net[:cut]),
                "oos_sharpe": _sharpe_np(net[cut:]),
            }
        )
    return pd.DataFrame(rows)


def overfitting_study(
    n_seeds: int = 200,
    n_bars: int = 2520,
    regimes: Sequence[str] = ("no_edge", "weak_trend"),
    grid: Sequence[tuple[int, int]] | None = None,
    split: float = 0.5,
    fee_bps: float = 1.0,
    execution_lag: int = 1,
    base_seed: int = 0,
) -> pd.DataFrame:
    """For each path, pick the grid point with the best in-sample Sharpe and record
    how it does out of sample."""
    grid = list(grid or ma_grid())
    rows = []
    for regime_name in regimes:
        reg = REGIMES[regime_name]
        for seed in _seeds(n_seeds, base_seed):
            g = grid_in_out_sample(reg.prices(n_bars, seed), grid, split, fee_bps, execution_lag)
            best = g.loc[g["is_sharpe"].idxmax()]
            rows.append(
                {
                    "regime": regime_name,
                    "seed": seed,
                    "best_fast": int(best["fast"]),
                    "best_slow": int(best["slow"]),
                    "best_is_sharpe": float(best["is_sharpe"]),
                    "best_oos_sharpe": float(best["oos_sharpe"]),
                    "median_is_sharpe": float(g["is_sharpe"].median()),
                    "median_oos_sharpe": float(g["oos_sharpe"].median()),
                    "best_oos_rank_pct": float((g["oos_sharpe"] < best["oos_sharpe"]).mean()),
                }
            )
    return pd.DataFrame(rows)


def summarise_overfitting(runs: pd.DataFrame, n_trials: int, n_is_obs: int) -> pd.DataFrame:
    g = runs.groupby("regime", sort=False)
    out = pd.DataFrame(
        {
            "mean_best_is_sharpe": g["best_is_sharpe"].mean(),
            "mean_best_oos_sharpe": g["best_oos_sharpe"].mean(),
            "mean_median_oos_sharpe": g["median_oos_sharpe"].mean(),
            "haircut": g["best_is_sharpe"].mean() - g["best_oos_sharpe"].mean(),
            "pct_best_is_gt_1": g["best_is_sharpe"].apply(lambda s: float((s > 1).mean())),
            "pct_best_oos_gt_0": g["best_oos_sharpe"].apply(lambda s: float((s > 0).mean())),
        }
    ).reset_index()
    out["null_expected_max_sharpe"] = expected_max_sharpe_under_null(n_trials, n_is_obs)
    return out


def overfitting_example(
    seed: int = 0,
    n_bars: int = 2520,
    regime: str = "no_edge",
    grid: Sequence[tuple[int, int]] | None = None,
    split: float = 0.5,
    fee_bps: float = 1.0,
    execution_lag: int = 1,
) -> tuple[pd.DataFrame, pd.Series, int]:
    """Detail for one path: the grid ranked by in-sample Sharpe, the chosen
    strategy's equity curve, and the split position."""
    grid = list(grid or ma_grid())
    close = REGIMES[regime].prices(n_bars, seed)
    g = grid_in_out_sample(close, grid, split, fee_bps, execution_lag)
    g = g.sort_values("is_sharpe", ascending=False).reset_index(drop=True)
    best = g.iloc[0]
    pos = ma_crossover(close, MACrossConfig(int(best["fast"]), int(best["slow"])))
    res = run_backtest(close, pos, fee_bps=fee_bps, execution_lag=execution_lag)
    cut = int(len(close) * split)
    # Cross-check the fast path against the engine on the chosen configuration.
    assert np.isclose(sharpe_ratio(res.strategy_returns.iloc[:cut]), best["is_sharpe"])
    return g, res.equity, cut


def regime_table() -> pd.DataFrame:
    return pd.DataFrame([asdict(r) for r in REGIMES.values()])
