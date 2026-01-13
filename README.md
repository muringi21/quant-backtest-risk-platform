## Quant Backtest & Risk Platform (v0.1)

### Run
```bash
pip install -e ".[dev]"
python -m qbrp.cli backtest --fast 20 --slow 50
pytest