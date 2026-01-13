from __future__ import annotations

from pathlib import Path

import typer

from .data import make_sample_prices
from .engine import run_backtest
from .strategies import MACrossConfig, ma_crossover


def main(
    fast: int = typer.Option(20, help="Fast moving average window."),
    slow: int = typer.Option(50, help="Slow moving average window."),
    fee_bps: float = typer.Option(1.0, "--fee-bps", help="Transaction fee in basis points."),
    plot: bool = typer.Option(True, "--plot/--no-plot", help="Save an equity curve plot."),
    plot_path: Path = typer.Option(
        Path("assets/equity.png"), "--plot-path", help="Where to save the equity plot."
    ),
):
    df = make_sample_prices()
    pos = ma_crossover(df["close"], MACrossConfig(fast=fast, slow=slow))
    res = run_backtest(df["close"], pos, fee_bps=fee_bps)

    typer.echo(f"Total return: {res.total_return:.2%}")
    typer.echo(f"Max drawdown: {res.max_drawdown:.2%}")
    typer.echo(f"Sharpe: {res.sharpe:.2f}")

    if plot:
        plot_path.parent.mkdir(parents=True, exist_ok=True)

        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 4))
        res.equity.plot()
        plt.title("Equity Curve")
        plt.xlabel("Date")
        plt.ylabel("Equity")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

        typer.echo(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    typer.run(main)