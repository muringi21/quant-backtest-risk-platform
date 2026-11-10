import numpy as np
import pandas as pd
import pytest

from qbrp.engine import run_backtest
from qbrp.experiments import (
    REGIMES,
    compare_strategies,
    fast_net_returns,
    fee_lag_sensitivity,
    grid_in_out_sample,
    ma_grid,
    overfitting_study,
    summarise_comparison,
    summarise_overfitting,
)
from qbrp.metrics import sharpe_ratio
from qbrp.report import ExperimentConfig, run_all
from qbrp.strategies import MACrossConfig, ma_crossover

SMALL = {"n_seeds": 3, "n_bars": 600}


@pytest.mark.parametrize("lag", [0, 1, 4])
def test_fast_path_matches_engine(lag):
    close = REGIMES["weak_trend"].prices(500, seed=1)
    pos = ma_crossover(close, MACrossConfig(10, 40))
    res = run_backtest(close, pos, fee_bps=3.0, execution_lag=lag)
    r = close.pct_change().fillna(0.0).to_numpy()
    fast = fast_net_returns(r, pos.to_numpy(), 3.0, lag)
    np.testing.assert_allclose(fast, res.strategy_returns.to_numpy(), atol=1e-15)


def test_grid_in_out_sample_matches_engine_for_one_pair():
    close = REGIMES["no_edge"].prices(800, seed=2)
    g = grid_in_out_sample(close, [(10, 50)], split=0.5, fee_bps=1.0, execution_lag=1)
    res = run_backtest(close, ma_crossover(close, MACrossConfig(10, 50)), 1.0, 1)
    assert g["is_sharpe"].iloc[0] == pytest.approx(sharpe_ratio(res.strategy_returns.iloc[:400]))
    assert g["oos_sharpe"].iloc[0] == pytest.approx(sharpe_ratio(res.strategy_returns.iloc[400:]))


def test_ma_grid_only_valid_pairs():
    grid = ma_grid()
    assert all(f < s for f, s in grid)
    assert len(grid) == len(set(grid)) == 174


def test_compare_is_deterministic_and_uses_common_paths():
    a = compare_strategies(**SMALL)
    b = compare_strategies(**SMALL)
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == 2 * 3 * 5
    s = summarise_comparison(a)
    assert s.loc[s["strategy"] == "buy_and_hold", "pct_beats_bh"].isna().all()


def test_sensitivity_costs_reduce_sharpe_monotonically():
    s = fee_lag_sensitivity(n_seeds=3, n_bars=600, lags=(0, 1), fees_bps=(0.0, 10.0, 50.0))
    for (_, _), grp in s.groupby(["strategy", "execution_lag"]):
        assert grp.sort_values("fee_bps")["mean_sharpe"].is_monotonic_decreasing


def test_overfitting_selection_bias_shows_up_even_in_a_small_run():
    grid = ma_grid(fast=(5, 10, 20), slow=(30, 60, 120))
    runs = overfitting_study(n_seeds=10, n_bars=1000, regimes=("no_edge",), grid=grid)
    summary = summarise_overfitting(runs, len(grid), 500)
    row = summary.iloc[0]
    # Picking the max over trials inflates the in-sample number above out-of-sample.
    assert row["mean_best_is_sharpe"] > row["mean_best_oos_sharpe"]
    assert (runs["best_is_sharpe"] >= runs["median_is_sharpe"]).all()


def test_report_is_byte_for_byte_reproducible(tmp_path):
    cfg = ExperimentConfig(**SMALL)
    run_all(tmp_path / "a", cfg, plots=False)
    run_all(tmp_path / "b", cfg, plots=False)
    for name in ["report.md", "compare_runs.csv", "sensitivity.csv", "overfit_runs.csv"]:
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()
    text = (tmp_path / "a" / "report.md").read_text()
    assert "simulated markets, not evidence about real ones" in text


def test_report_writes_figures_when_matplotlib_available(tmp_path):
    pytest.importorskip("matplotlib")
    out = run_all(tmp_path, ExperimentConfig(**SMALL), plots=True)
    assert set(out["figures"]) == {
        "compare_sharpe.png",
        "sensitivity.png",
        "overfit_example.png",
        "overfit_scatter.png",
    }
    assert all((tmp_path / "figures" / f).stat().st_size > 0 for f in out["figures"])
