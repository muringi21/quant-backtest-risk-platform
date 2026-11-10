import numpy as np
import pandas as pd
import pytest

from qbrp.data import load_prices_csv, make_sample_prices, simulate_log_returns
from qbrp.errors import BacktestInputError


def test_same_seed_same_frame_including_index():
    a = make_sample_prices(n=100, seed=42)
    b = make_sample_prices(n=100, seed=42)
    pd.testing.assert_frame_equal(a, b)
    assert a.index[0] == pd.Timestamp("2020-01-01")


def test_different_seeds_differ():
    assert not make_sample_prices(seed=1)["close"].equals(make_sample_prices(seed=2)["close"])


def test_shape_and_positivity():
    df = make_sample_prices(n=756)
    assert list(df.columns) == ["close"]
    assert len(df) == 756
    assert (df["close"] > 0).all()
    assert df.index.is_monotonic_increasing


def _autocorr(x, k):
    return np.corrcoef(x[:-k], x[k:])[0, 1]


@pytest.mark.parametrize("share, rho", [(0.0, 0.98), (0.1, 0.9), (0.3, 0.5)])
def test_moments_match_parameters(share, rho):
    r = simulate_log_returns(
        400_000, seed=0, drift=0.0005, vol=0.01, trend_share=share, trend_persistence=rho
    )
    assert r.mean() == pytest.approx(0.0005, abs=1e-4)
    assert r.std() == pytest.approx(0.01, rel=0.02)
    for k in (1, 5):
        assert _autocorr(r, k) == pytest.approx(share * rho**k, abs=0.01)


def test_random_walk_default_has_no_autocorrelation():
    r = simulate_log_returns(200_000, seed=1)
    assert abs(_autocorr(r, 1)) < 0.01


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"trend_share": 1.0}, "trend_share"),
        ({"trend_persistence": 1.0}, "trend_persistence"),
        ({"vol": -0.1}, "vol"),
        ({"n": 1}, "n"),
        ({"seed": -3}, "seed"),
        ({"start": "not-a-date"}, "start"),
    ],
)
def test_invalid_generator_args(kwargs, match):
    with pytest.raises(BacktestInputError, match=match):
        make_sample_prices(**kwargs)


def test_load_csv_sorts_by_date_and_is_case_insensitive(tmp_path):
    f = tmp_path / "px.csv"
    f.write_text("Date,Close\n2024-01-03,102\n2024-01-01,100\n2024-01-02,101\n")
    df = load_prices_csv(f)
    assert df["close"].tolist() == [100, 101, 102]
    assert isinstance(df.index, pd.DatetimeIndex)


def test_load_csv_without_dates(tmp_path):
    f = tmp_path / "px.csv"
    f.write_text("adj_close\n100\n101\n")
    df = load_prices_csv(f, column="adj_close", date_column=None)
    assert df["close"].tolist() == [100, 101]


@pytest.mark.parametrize(
    "body, match",
    [
        ("date,close\n2024-01-01,100\n2024-01-02,\n", "NaN"),
        ("date,close\n2024-01-01,100\n2024-01-02,abc\n", "non-numeric"),
        ("date,close\n2024-01-01,100\n2024-01-02,-5\n", "strictly positive"),
        ("date,close\n2024-01-01,100\n2024-01-01,101\n", "duplicate"),
        ("date,price\n2024-01-01,100\n2024-01-02,101\n", "not found"),
        ("date,close\nyesterday,100\n2024-01-02,101\n", "dates"),
    ],
)
def test_load_csv_rejects_bad_data(tmp_path, body, match):
    f = tmp_path / "bad.csv"
    f.write_text(body)
    with pytest.raises(BacktestInputError, match=match):
        load_prices_csv(f)


def test_missing_file(tmp_path):
    with pytest.raises(BacktestInputError, match="not found"):
        load_prices_csv(tmp_path / "nope.csv")
