import numpy as np
import pandas as pd

def daily_returns(prices: pd.Series) -> pd.Series:
    return prices.pct_change().fillna(0.0).rename("asset_ret")

def equity_curve(strategy_returns: pd.Series, start: float = 1.0) -> pd.Series:
    return (start * (1.0 + strategy_returns).cumprod()).rename("equity")

def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    return float(dd.min())

def sharpe_ratio(strategy_returns: pd.Series, periods_per_year: int = 252) -> float:
    r = strategy_returns.dropna()
    std = float(r.std(ddof=1))
    if std < 1e-12:
        return 0.0
    return float(np.sqrt(periods_per_year) * float(r.mean()) / std)