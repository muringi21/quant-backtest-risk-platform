from dataclasses import dataclass
import pandas as pd

@dataclass(frozen=True)
class MACrossConfig:
    fast: int = 20
    slow: int = 50

def ma_crossover(prices: pd.Series, cfg: MACrossConfig) -> pd.Series:
    fast_ma = prices.rolling(cfg.fast).mean()
    slow_ma = prices.rolling(cfg.slow).mean()
    signal = (fast_ma > slow_ma).astype(int) - (fast_ma < slow_ma).astype(int)
    return signal.fillna(0).rename("position")