from dataclasses import dataclass
import pandas as pd

from .metrics import daily_returns, equity_curve, max_drawdown, sharpe_ratio

@dataclass
class BacktestResult:
    equity: pd.Series
    strategy_returns: pd.Series
    total_return: float
    max_drawdown: float
    sharpe: float

def run_backtest(close: pd.Series, position: pd.Series, fee_bps: float = 1.0) -> BacktestResult:
    asset_ret = daily_returns(close)

    pos = position.reindex(close.index).fillna(0.0)
    pos_lag = pos.shift(1).fillna(0.0)  # avoids lookahead bias

    turnover = (pos - pos_lag).abs()
    fee = (fee_bps / 10_000.0) * turnover  # bps -> fraction

    strat_ret = (pos_lag * asset_ret) - fee
    strat_ret = strat_ret.rename("strategy_ret")

    eq = equity_curve(strat_ret, start=1.0)

    return BacktestResult(
        equity=eq,
        strategy_returns=strat_ret,
        total_return=float(eq.iloc[-1] - 1.0),
        max_drawdown=max_drawdown(eq),
        sharpe=sharpe_ratio(strat_ret),
    )