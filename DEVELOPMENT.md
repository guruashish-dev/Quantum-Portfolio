# Development notes

## Commands executed

- `python -m pip install -r requirements.txt` (Python 3.13.5 environment)
- `python run_pipeline.py --mode sample --epochs 1` (end-to-end run)
- `pytest -q` (verification)
- Dashboard smoke check: `streamlit run app/streamlit_app.py --server.headless true`

## Dependencies

Declared in `requirements.txt`: pandas, numpy, scikit-learn, scipy, yfinance, matplotlib, plotly, streamlit, pennylane, pennylane-lightning, joblib, pytest. PennyLane 0.45.1 and lightning simulator ran locally.

## Reproduce

Create a Python 3.11+ environment, install requirements, then run `python run_pipeline.py --mode sample --epochs 1`. The included data is deterministic synthetic OHLCV for offline software demonstration; it is not reported as observed historical market data. Model outputs are generated from actual execution and stored under `results/`. Increase to 20 epochs for the configured research run.

For current market prices run `python run_pipeline.py --mode market --epochs 1`. yfinance downloads are cached under `data/raw/`; stale/cache refresh currently requires removing the relevant cached CSV. Dashboard: `streamlit run app/streamlit_app.py`.

## Known limitations

This is an academic prototype. It omits transaction costs, taxes, dividends, slippage and execution constraints. The sample signal is deliberately minimal, backtest benchmark returns are daily equal-weight and buy-and-hold series, and a one-epoch smoke run is only for runtime verification. No quantum advantage is claimed. Market downloads depend on network availability and provider behavior.
