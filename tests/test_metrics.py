import pandas as pd
from qbrp.metrics import max_drawdown, sharpe_ratio

def test_max_drawdown_non_positive():
    eq = pd.Series([1.0, 1.1, 1.05, 1.2, 1.15])
    assert max_drawdown(eq) <= 0.0

def test_sharpe_zero_when_no_variance():
    r = pd.Series([0.01] * 10)
    assert sharpe_ratio(r) == 0.0