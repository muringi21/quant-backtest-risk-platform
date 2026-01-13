import numpy as np
import pandas as pd

def make_sample_prices(n: int = 756, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(loc=0.0003, scale=0.01, size=n)
    price = 100 * np.exp(np.cumsum(rets))
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    return pd.DataFrame({"close": price}, index=idx)