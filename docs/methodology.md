# Methodology, assumptions and limitations

This document says exactly what the engine computes, which simplifications it
makes, and what the results can and cannot tell you. If a number produced by
`qbrp` surprises you, the answer should be somewhere in here.

## Scope

`qbrp` backtests **one asset at a time** using **close prices only**. A
strategy is a function that turns a price series into a target exposure in
`[-1, 1]` per bar. There is no order book, no portfolio of several assets, no
intraday data, no live trading and no external data feed. It is a research
tool for reasoning about simple rules, not a trading system.

## Timing model

For bar `t` (e.g. one trading day), with close price `p[t]`:

| Symbol | Meaning |
|---|---|
| `r[t] = p[t]/p[t-1] - 1` | asset simple return; `r[0] = 0` |
| `position[t]` | target exposure decided at the close of bar `t`, using prices up to and including `p[t]` |
| `L` (`execution_lag`) | number of bars between the decision and the trade |
| `held[t] = position[t-L]` | exposure held after the close of bar `t` (0 before the first signal) |
| `gross[t] = held[t-1] * r[t]` | return earned on the exposure carried into bar `t` |
| `cost[t] = fee_bps/10,000 * abs(held[t] - held[t-1])` | cost of trading at the close of bar `t` |
| `net[t] = gross[t] - cost[t]` | strategy return; equity is `prod(1 + net)` |

**Why this has no look-ahead.** The return paid at bar `t` uses `held[t-1]`,
which depends on prices up to `t-1-L`. A strategy that only looks backwards
therefore never sees the return it is paid on. `tests/test_engine.py` checks
this directly: rewriting every price after bar `k` leaves all strategy returns
up to bar `k` unchanged, for several lags.

**What the engine cannot catch.** If a strategy function itself uses future
data (for example `prices.shift(-1)`), the engine will faithfully report an
impossible result. A test (`test_peeking_strategy_is_detected...`) shows a
peeking signal producing a Sharpe above 10 - that is what a look-ahead bug
looks like. All bundled strategies are tested to depend only on past prices.

**`execution_lag=0` vs `1`.** Lag 0 trades at the same close used to compute
the signal. That assumes you can observe the close and trade at it, which is
optimistic. The default since v0.2 is lag 1 (trade at the next close). v0.1
behaved like lag 0.

## Costs

* One proportional cost, `fee_bps`, per unit of turnover. Going from flat to
  fully long is 1 unit; flipping from long to short is 2 units.
* It stands in for commission, half-spread and slippage together. There is no
  separate market-impact model, no minimum ticket fee and no size dependence.
* Costs are charged at entry; no cost is charged to close the final position.
* `total_cost` in the output is the sum of per-bar cost fractions. Because each
  is a fraction of equity at that time, it is an approximation of the share of
  starting capital spent on costs, not an exact figure.

## Other simplifications

* **Fills at the close price.** No partial fills, no gaps between signal and
  fill other than the integer lag.
* **Shorting is free and unlimited.** No borrow fee, no recall risk, no margin
  calls. A short position earns `-r[t]` per bar, which assumes daily
  rebalancing to constant exposure.
* **No financing.** Cash earns nothing and leverage is not allowed
  (`|position| <= 1`). The risk-free rate is zero in all ratios.
* **Daily rebalancing to target.** Exposure is reset every bar; with a
  constant target this costs nothing because turnover is measured on the
  target, not on drift in the position's value.
* **No dividends, splits or corporate actions** unless they are already in
  the price series you supply (use adjusted closes from real data).

## Metrics

| Metric | Definition |
|---|---|
| Total return | `equity[-1] - 1` |
| CAGR | `equity[-1] ** (periods_per_year / n_bars) - 1` |
| Volatility | sample std of `net`, times `sqrt(periods_per_year)` |
| Sharpe | `sqrt(periods_per_year) * mean(net) / std(net)`, risk-free rate 0; **NaN** if the std is zero or there is one observation |
| Sharpe s.e. | Lo (2002) i.i.d. approximation `sqrt((1 + s^2/2) / T)` for per-period Sharpe `s`, annualised |
| Max drawdown | most negative value of `equity / running_max(equity) - 1` |
| Position changes | number of bars with non-zero turnover |
| Turnover/yr | total turnover divided by years |
| Exposure | mean of `abs(held)` |

Bar 0 has no asset return (`r[0] = 0`) and is still counted as one period;
over hundreds of bars this changes annualised numbers negligibly.

Sharpe returned `0.0` for a constant series in v0.1. That looked like a real
measurement ("zero risk-adjusted return") when the ratio is actually undefined,
so it now returns NaN and the CLI prints `n/a`.

## Statistical limitations

* **One backtest is one sample.** With daily data, the standard error of an
  annualised Sharpe is roughly `1 / sqrt(years)`: about 0.58 for 3 years and
  0.32 for 10 years. A 10-year Sharpe of 0.6 is not reliably different from 0.
  The CLI prints this standard error next to the Sharpe for that reason.
* **The standard error is optimistic.** It assumes independent, normally
  distributed returns. Fat tails, volatility clustering and autocorrelation in
  real returns all make the true uncertainty larger.
* **Trying many parameters inflates the best result.** If you test `N` variants
  and report the best, the expected best Sharpe is positive even when every
  variant is worthless. `metrics.expected_max_sharpe_under_null` gives the
  Bailey & Lopez de Prado (2014) estimate for `N` independent trials; correlated
  trials (like neighbouring moving-average windows) behave like fewer
  independent ones. See `docs/analysis.md`, section 3.
* **Synthetic data is not market data.** The generator produces Gaussian
  returns with constant volatility and, optionally, a hidden trend whose
  strength we chose. That makes it ideal for checking *whether the method can
  recover a known truth*, and useless for claiming that a strategy works on
  real markets.

## Synthetic price model

`make_sample_prices` draws daily log returns

```
r[t] = drift + x[t] + e[t]
x[t] = persistence * x[t-1] + u[t]     (hidden trend, stationary AR(1))
e[t] ~ N(0, (1 - trend_share) * vol^2)
```

with the variance of `u` chosen so that `x` contributes `trend_share * vol^2`
and the total standard deviation of `r` is exactly `vol`. Return
autocorrelation at lag `k` is `trend_share * persistence^k`, so the
`weak_trend` regime used in the experiments (share 0.02, persistence 0.98) has
a lag-1 autocorrelation of about 0.02, which would be hard to detect with a
standard test on a few years of data, yet it is enough to give trend rules a
real edge. With `trend_share=0` (the default) the series is a geometric random
walk with drift. The calendar starts at a fixed date, so a given seed always
produces the same DataFrame.

## Input validation

All public functions validate inputs and raise `qbrp.errors.BacktestInputError`
(a subclass of `ValueError`) naming the bad argument. Examples: non-positive or
NaN prices, unsorted or duplicate dates, a position series whose index does not
match the prices exactly (v0.1 silently reindexed and filled with 0), exposure
outside `[-1, 1]`, negative fees, a lag longer than the data, `fast >= slow`,
and too few prices for the chosen windows. The CSV loader refuses to drop or
fill bad rows: it is better to fail than to backtest on silently repaired data.
