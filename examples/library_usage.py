"""Use qbrp as a library: compare a rule with buy-and-hold on the same prices.

Run with:  python examples/library_usage.py
"""

from qbrp.data import make_sample_prices
from qbrp.engine import run_backtest
from qbrp.strategies import MACrossConfig, buy_and_hold, ma_crossover

close = make_sample_prices(n=2520, seed=1, trend_share=0.02)["close"]

for name, position in [
    ("Buy and hold", buy_and_hold(close)),
    ("MA 20/50 long-short", ma_crossover(close, MACrossConfig(20, 50))),
    ("MA 20/50 long-only", ma_crossover(close, MACrossConfig(20, 50, long_only=True))),
]:
    for fee in (1.0, 20.0):
        res = run_backtest(close, position, fee_bps=fee, execution_lag=1)
        print(
            f"{name:<22} fee {fee:>4.0f} bp  Sharpe {res.sharpe:5.2f} +/- {res.sharpe_se:.2f}  "
            f"CAGR {res.cagr:6.1%}  max DD {res.max_drawdown:6.1%}  "
            f"turnover {res.annual_turnover:4.1f}x/yr"
        )
