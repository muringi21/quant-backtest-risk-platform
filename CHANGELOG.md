# Changelog

## 0.2.0

A rewrite focused on correctness, transparency and reproducibility. Several
default numbers change; the reasons are listed below.

### Changed (affects results)
- **Default execution lag is now 1 bar.** v0.1 traded at the same close that
  produced the signal; that is still available as `execution_lag=0` / `--lag 0`.
- **Sharpe ratio returns NaN when undefined** (zero variance or one
  observation). v0.1 returned 0.0, which looked like a real measurement.
- **Synthetic prices start on a fixed date** (2020-01-01) rather than ending
  today, so the same seed produces an identical DataFrame on any day. Prices
  for a given seed are unchanged.
- **Positions must share the price index exactly and lie in [-1, 1].** v0.1
  silently reindexed and filled missing positions with 0.

### Added
- Input validation everywhere, with `BacktestInputError` naming the bad argument.
- Metrics: CAGR, annualised volatility, drawdown series, Sharpe standard
  error, expected max Sharpe under the null, turnover, exposure, costs.
- Strategies: long-only MA crossover, time-series momentum, buy-and-hold.
- Synthetic generator: optional hidden trend with known strength.
- CSV price loader.
- CLI: `qbrp backtest` with strategy choice, CSV input, lag, JSON output and
  per-bar export; `qbrp experiment all`; `--version`; clean errors with exit code 2.
- Reproducible experiments, generated report and figures under `docs/results`.
- Documentation: methodology and limitations, and a written analysis.
- Tests: from 3 to over 100.

### Fixed
- Committed `__pycache__` and `egg-info` files removed; `.gitignore` added.
- CI lint step was failing; it now passes and runs on several Python versions.

### Migration
- `python -m qbrp.cli --fast 20 --slow 50` becomes `qbrp backtest --fast 20 --slow 50`.
- To reproduce v0.1 numbers, add `--lag 0`, and note Sharpe of a flat
  strategy now prints `n/a`.

## 0.1.0
- Synthetic prices, MA crossover, backtest with fees in bps, total return,
  max drawdown and Sharpe, CLI and CI.
