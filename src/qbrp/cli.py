"""Command-line interface: ``qbrp backtest`` and ``qbrp experiment``."""

from __future__ import annotations

import json
import math
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from . import __version__
from .data import load_prices_csv, make_sample_prices
from .engine import BacktestResult, run_backtest
from .errors import BacktestInputError
from .strategies import MACrossConfig, MomentumConfig, buy_and_hold, ma_crossover, momentum

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Single-asset backtester with explicit execution timing and transaction costs.",
)
experiment_app = typer.Typer(no_args_is_help=True, help="Reproducible synthetic experiments.")
app.add_typer(experiment_app, name="experiment")


class Strategy(StrEnum):
    ma = "ma"
    ma_long = "ma-long"
    momentum = "momentum"
    buy_hold = "buy-hold"


def _fail(message: str) -> None:
    typer.secho(f"Error: {message}", err=True, fg=typer.colors.RED)
    raise typer.Exit(code=2)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"qbrp {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version."),
    ] = False,
) -> None:
    """qbrp - a small, honest backtesting engine."""


def _fmt_pct(x: float) -> str:
    return "n/a" if math.isnan(x) else f"{x:.2%}"


def _fmt_num(x: float) -> str:
    return "n/a" if math.isnan(x) else f"{x:.2f}"


def _print_result(res: BacktestResult, title: str, n_bars: int) -> None:
    se = res.sharpe_se
    sharpe = _fmt_num(res.sharpe)
    if not math.isnan(se):
        sharpe += f"  (+/- {se:.2f} s.e.)"
    lines = [
        title,
        f"  Bars: {n_bars}   Fee: {res.fee_bps:g} bp/turnover   Execution lag: "
        f"{res.execution_lag} bar(s)",
        f"  Total return:     {_fmt_pct(res.total_return)}",
        f"  CAGR:             {_fmt_pct(res.cagr)}",
        f"  Volatility (ann): {_fmt_pct(res.ann_vol)}",
        f"  Sharpe (rf=0):    {sharpe}",
        f"  Max drawdown:     {_fmt_pct(res.max_drawdown)}",
        f"  Position changes: {res.n_trades}   Turnover/yr: {res.annual_turnover:.1f}x   "
        f"Avg |exposure|: {res.exposure:.0%}",
        f"  Costs (sum):      {res.total_cost:.2%} of equity, summed over bars",
    ]
    typer.echo("\n".join(lines))


@app.command()
def backtest(
    strategy: Annotated[
        Strategy, typer.Option(help="Signal: MA crossover, long-only MA, momentum, buy-and-hold.")
    ] = Strategy.ma,
    fast: Annotated[int, typer.Option(help="Fast moving-average window (bars).")] = 20,
    slow: Annotated[int, typer.Option(help="Slow moving-average window (bars).")] = 50,
    lookback: Annotated[int, typer.Option(help="Momentum lookback (bars).")] = 126,
    fee_bps: Annotated[float, typer.Option(help="Cost in bp per unit of turnover.")] = 1.0,
    lag: Annotated[
        int, typer.Option(help="Bars between signal and execution (0 = same close).")
    ] = 1,
    csv: Annotated[
        Path | None, typer.Option(help="Read prices from this CSV instead of simulating.")
    ] = None,
    column: Annotated[str, typer.Option(help="Price column in the CSV.")] = "close",
    date_column: Annotated[str, typer.Option(help="Date column in the CSV, if any.")] = "date",
    n: Annotated[int, typer.Option("--bars", help="Synthetic: number of bars.")] = 756,
    seed: Annotated[int, typer.Option(help="Synthetic: random seed.")] = 7,
    drift: Annotated[float, typer.Option(help="Synthetic: mean daily log return.")] = 0.0003,
    vol: Annotated[float, typer.Option(help="Synthetic: daily return std dev.")] = 0.01,
    trend_share: Annotated[
        float, typer.Option(help="Synthetic: share of variance from a hidden trend (0 = none).")
    ] = 0.0,
    periods_per_year: Annotated[int, typer.Option(help="For annualising metrics.")] = 252,
    as_json: Annotated[bool, typer.Option("--json", help="Print metrics as JSON.")] = False,
    equity_out: Annotated[
        Path | None, typer.Option(help="Write per-bar positions, returns and equity to CSV.")
    ] = None,
) -> None:
    """Run one backtest on synthetic or CSV prices and print its metrics."""
    try:
        if csv is not None:
            close = load_prices_csv(csv, column=column, date_column=date_column)["close"]
            source = f"{csv.name}:{column}"
        else:
            close = make_sample_prices(
                n=n, seed=seed, drift=drift, vol=vol, trend_share=trend_share
            )["close"]
            source = f"synthetic (seed {seed})"

        if strategy in (Strategy.ma, Strategy.ma_long):
            cfg = MACrossConfig(fast, slow, long_only=strategy is Strategy.ma_long)
            pos = ma_crossover(close, cfg)
            name = f"MA {fast}/{slow} {'long-only' if cfg.long_only else 'long-short'}"
        elif strategy is Strategy.momentum:
            pos = momentum(close, MomentumConfig(lookback))
            name = f"Momentum {lookback} long-short"
        else:
            pos = buy_and_hold(close)
            name = "Buy and hold"

        res = run_backtest(
            close, pos, fee_bps=fee_bps, execution_lag=lag, periods_per_year=periods_per_year
        )
    except BacktestInputError as exc:
        _fail(str(exc))
        return

    if equity_out is not None:
        frame = pd.DataFrame(
            {
                "close": close,
                "target_position": pos,
                "held_position": res.held_position,
                "gross_ret": res.gross_returns,
                "cost": res.costs,
                "strategy_ret": res.strategy_returns,
                "equity": res.equity,
            }
        )
        frame.to_csv(equity_out, index_label="date")

    if as_json:
        typer.echo(json.dumps({"strategy": name, "source": source, **res.summary()}, indent=2))
    else:
        _print_result(res, f"{name} on {source}", len(close))


@experiment_app.command("all")
def experiment_all(
    out: Annotated[Path, typer.Option(help="Output directory.")] = Path("results"),
    seeds: Annotated[int, typer.Option(help="Number of simulated paths per study.")] = 200,
    n_bars: Annotated[int, typer.Option(help="Bars per path.")] = 2520,
    base_seed: Annotated[int, typer.Option(help="First seed.")] = 0,
    plots: Annotated[bool, typer.Option(help="Write PNG figures (needs matplotlib).")] = True,
) -> None:
    """Run the strategy comparison, fee/lag sensitivity and overfitting studies."""
    from .report import ExperimentConfig, run_all

    try:
        cfg = ExperimentConfig(n_seeds=seeds, n_bars=n_bars, base_seed=base_seed)
        result = run_all(out, cfg, plots=plots)
    except BacktestInputError as exc:
        _fail(str(exc))
        return
    typer.echo(f"Wrote results to {out}/ (report.md, CSVs)")
    if plots and not result["figures"]:
        typer.echo("matplotlib not installed; skipped figures (pip install -e '.[plots]').")


if __name__ == "__main__":
    app()
