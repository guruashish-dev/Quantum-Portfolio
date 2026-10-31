# Q-Portfolio — Hybrid Quantum Machine Learning for Portfolio Management

Academic proof of concept evaluating whether a small PennyLane variational classifier can predict next-period direction and inform a constrained portfolio. It makes no claim of quantum advantage and is not a trading system.

## Method

Daily asset features are momentum, rolling volatility, RSI, moving-average ratio, rolling mean return and volume change. Labels are next-period return > 0; rows without a known future label are dropped. Chronological train/validation/test fractions default to 70/15/15 because random splitting leaks temporal structure and future regimes into training. Logistic regression is the baseline. Four scaled features are angle-encoded into four qubits, followed by trainable RY/RZ rotations, CNOT entanglement and Pauli-Z expectation; a classical optimizer trains circuit parameters.

The optimizer maximizes `wᵀμ − λwᵀΣw`, subject to weights summing to one and bounded by a configurable cap. Portfolio return is `rₚ,t = wᵀrₜ`; variance is `wᵀΣw`; Sharpe is `(Rannual − Rf)/σannual`. Walk-forward weights use only information available at rebalance time. Sample data is deterministic synthetic OHLCV for software demonstrations, not claimed to be historical market data. Market mode uses cached yfinance downloads.

## Install and run

Python 3.11+ recommended. `python -m venv .venv`, activate, `pip install -r requirements.txt`. Run `python run_pipeline.py --mode sample --epochs 20`; market mode: `--mode market`. Dashboard: `streamlit run app/streamlit_app.py`. Module commands: `python -m src.data.downloader`, `python -m src.models.classical_model`, `python -m src.models.quantum_model`, `python -m src.portfolio.backtester`. Run checks using `pytest`.

Outputs are computed at runtime in `results/`: model metrics, trained models/parameters, allocation, backtest series and plots. No metrics are asserted in advance. The dashboard runs the pipeline on demand and also displays generated artifacts. Circuit supports lightning.qubit where available and falls back to default.qubit.

## Limitations and research

Synthetic sample prices are for reproducibility only; costs, taxes, liquidity, corporate actions and realistic execution are omitted. Short samples and small circuits limit inference. Predictions are not investment advice. Future work includes more robust walk-forward validation, transaction costs, calibrated uncertainty, larger asset universes and statistical significance tests. Research question: can a compact hybrid variational classifier produce useful out-of-sample signals compared with logistic regression and simple benchmarks? Quantum encoding maps normalized feature values to gate rotation angles; variational gates are learned parameters optimized classically from measured expectations.
