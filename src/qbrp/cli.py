import typer

from .data import make_sample_prices
from .engine import run_backtest
from .strategies import MACrossConfig, ma_crossover

app = typer.Typer(no_args_is_help=True)

@app.command()
def backtest(fast: int = 20, slow: int = 50, fee_bps: float = 1.0):
    df = make_sample_prices()
    pos = ma_crossover(df["close"], MACrossConfig(fast=fast, slow=slow))
    res = run_backtest(df["close"], pos, fee_bps=fee_bps)

    typer.echo(f"Total return: {res.total_return:.2%}")
    typer.echo(f"Max drawdown: {res.max_drawdown:.2%}")
    typer.echo(f"Sharpe: {res.sharpe:.2f}")

if __name__ == "__main__":
    app()