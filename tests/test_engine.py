from qbrp.data import make_sample_prices
from qbrp.strategies import MACrossConfig, ma_crossover
from qbrp.engine import run_backtest

def test_backtest_runs():
    df = make_sample_prices(n=200)
    pos = ma_crossover(df["close"], MACrossConfig(5, 20))
    res = run_backtest(df["close"], pos)
    assert len(res.equity) == len(df)
    assert res.equity.isna().sum() == 0