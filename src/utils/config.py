from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
ASSETS=['AAPL','MSFT','GOOGL','AMZN','NVDA']
FEATURES=['momentum','volatility','rsi','moving_average_ratio','rolling_mean_return','volume_change']
QUANTUM_FEATURES=['momentum','volatility','rsi','moving_average_ratio']
SEED=42
