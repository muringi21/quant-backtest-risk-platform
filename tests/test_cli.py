import json

import pandas as pd
import pytest
from typer.testing import CliRunner

from qbrp.cli import app
from qbrp.data import make_sample_prices
from qbrp.engine import run_backtest
from qbrp.strategies import MACrossConfig, ma_crossover

runner = CliRunner()


def test_default_backtest_prints_metrics():
    result = runner.invoke(app, ["backtest"])
    assert result.exit_code == 0, result.output
    for label in ("Total return", "Sharpe", "Max drawdown", "Execution lag: 1"):
        assert label in result.output


def test_json_output_matches_library():
    args = ["backtest", "--json", "--fast", "10", "--slow", "30", "--lag", "0"]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    out = json.loads(result.output)
    close = make_sample_prices()["close"]
    lib = run_backtest(close, ma_crossover(close, MACrossConfig(10, 30)), 1.0, 0)
    assert out["sharpe"] == pytest.approx(lib.sharpe)
    assert out["total_return"] == pytest.approx(lib.total_return)
    assert out["execution_lag"] == 0


@pytest.mark.parametrize("strategy", ["ma", "ma-long", "momentum", "buy-hold"])
def test_every_strategy_runs(strategy):
    result = runner.invoke(app, ["backtest", "--strategy", strategy, "--json"])
    assert result.exit_code == 0, result.output
    assert "sharpe" in json.loads(result.output)


@pytest.mark.parametrize(
    "args, message",
    [
        (["--fast", "50", "--slow", "20"], "must be shorter than slow"),
        (["--fee-bps", "-1"], "fee_bps"),
        (["--lag", "-1"], "execution_lag"),
        (["--bars", "30"], "at least slow=50"),
        (["--csv", "does-not-exist.csv"], "not found"),
    ],
)
def test_bad_input_gives_clean_error_and_exit_code_2(args, message):
    result = runner.invoke(app, ["backtest", *args])
    assert result.exit_code == 2
    assert message in result.output
    assert "Traceback" not in result.output


def test_csv_input_and_equity_export(tmp_path):
    df = make_sample_prices(n=120, seed=3)
    src = tmp_path / "prices.csv"
    df.to_csv(src, index_label="date")
    dest = tmp_path / "equity.csv"
    result = runner.invoke(
        app,
        ["backtest", "--csv", str(src), "--fast", "5", "--slow", "20", "--equity-out", str(dest)],
    )
    assert result.exit_code == 0, result.output
    assert "prices.csv:close" in result.output
    exported = pd.read_csv(dest, index_col="date", parse_dates=True)
    assert len(exported) == 120
    assert {"held_position", "strategy_ret", "equity", "cost"} <= set(exported.columns)


def test_version_flag():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.output.startswith("qbrp ")


def test_experiment_all_small_run(tmp_path):
    result = runner.invoke(
        app,
        ["experiment", "all", "--out", str(tmp_path), "--seeds", "2", "--n-bars", "600",
         "--no-plots"],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "compare_summary.csv").exists()
