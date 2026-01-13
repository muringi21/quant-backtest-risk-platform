# Quant Backtest & Risk Platform (v0.1)

A minimal single-asset backtesting + risk-metrics toolkit.

## Features
- Synthetic price data generator
- MA crossover strategy
- Backtest engine with simple transaction costs (bps)
- Metrics: total return, max drawdown, Sharpe
- CLI interface
- Unit tests + GitHub Actions CI

## Quickstart

```bash
pip install -e ".[dev]"
pytest
python -m qbrp.cli --fast 20 --slow 50 --fee-bps 1.0
