from pathlib import Path
p=Path('.')
files={
'README.md':'''# Q-Portfolio — Hybrid Quantum Machine Learning for Portfolio Management

Academic proof of concept evaluating whether a small PennyLane variational classifier can predict next-period direction and inform a constrained portfolio. It makes no claim of quantum advantage and is not a trading system.

## Method

Daily asset features are momentum, rolling volatility, RSI, moving-average ratio, rolling mean return and volume change. Labels are next-period return > 0; rows without a known future label are dropped. Chronological train/validation/test fractions default to 70/15/15 because random splitting leaks temporal structure and future regimes into training. Logistic regression is the baseline. Four scaled features are angle-encoded into four qubits, followed by trainable RY/RZ rotations, CNOT entanglement and Pauli-Z expectation; a classical optimizer trains circuit parameters.

The optimizer maximizes `wᵀμ − λwᵀΣw`, subject to weights summing to one and bounded by a configurable cap. Portfolio return is `rₚ,t = wᵀrₜ`; variance is `wᵀΣw`; Sharpe is `(Rannual − Rf)/σannual`. Walk-forward weights use only information available at rebalance time. Sample data is deterministic synthetic OHLCV for software demonstrations, not claimed to be historical market data. Market mode uses cached yfinance downloads.

## Install and run

Python 3.11+ recommended. `python -m venv .venv`, activate, `pip install -r requirements.txt`. Run `python run_pipeline.py --mode sample --epochs 20`; market mode: `--mode market`. Dashboard: `streamlit run app/streamlit_app.py`. Module commands: `python -m src.data.downloader`, `python -m src.models.classical_model`, `python -m src.models.quantum_model`, `python -m src.portfolio.backtester`. Run checks using `pytest`.

Outputs are computed at runtime in `results/`: model metrics, trained models/parameters, allocation, backtest series and plots. No metrics are asserted in advance. The dashboard runs the pipeline on demand and also displays generated artifacts. Circuit supports lightning.qubit where available and falls back to default.qubit.

## Limitations and research

Synthetic sample prices are for reproducibility only; costs, taxes, liquidity, corporate actions and realistic execution are omitted. Short samples and small circuits limit inference. Predictions are not investment advice. Future work includes more robust walk-forward validation, transaction costs, calibrated uncertainty, larger asset universes and statistical significance tests. Research question: can a compact hybrid variational classifier produce useful out-of-sample signals compared with logistic regression and simple benchmarks? Quantum encoding maps normalized feature values to gate rotation angles; variational gates are learned parameters optimized classically from measured expectations.
''',
'requirements.txt':'numpy\npandas\nscikit-learn\nscipy\nyfinance\nmatplotlib\nplotly\nstreamlit\npennylane\npennylane-lightning\njoblib\npytest\n',
'.gitignore':'__pycache__/\n.venv/\ndata/raw/*\nresults/*\nmodels/*\n!.gitkeep\n',
'src/utils/config.py':'''from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
ASSETS=['AAPL','MSFT','GOOGL','AMZN','NVDA']
FEATURES=['momentum','volatility','rsi','moving_average_ratio','rolling_mean_return','volume_change']
QUANTUM_FEATURES=['momentum','volatility','rsi','moving_average_ratio']
SEED=42
''',
'src/data/downloader.py':'''"""Cached yfinance OHLCV download helper."""
from pathlib import Path
import pandas as pd

def download(tickers, period='5y', cache=Path('data/raw')):
    import yfinance as yf
    cache.mkdir(parents=True,exist_ok=True); out={}
    for ticker in tickers:
        f=cache/f'{ticker.replace(".","_")}.csv'
        if f.exists():
            try: out[ticker]=pd.read_csv(f,index_col=0,parse_dates=True); continue
            except Exception: pass
        try:
            df=yf.download(ticker,period=period,auto_adjust=True,progress=False)
            if isinstance(df.columns,pd.MultiIndex): df.columns=df.columns.get_level_values(0)
            if not df.empty: df.to_csv(f); out[ticker]=df
        except Exception as e: print(f'{ticker}: download failed: {e}')
    return out
if __name__=='__main__':
    from src.utils.config import ASSETS
    print({k:len(v) for k,v in download(ASSETS).items()})
''',
'src/data/loader.py':'''"""Load sample or cached/downloaded market OHLCV."""
from pathlib import Path
import pandas as pd
from src.utils.config import ASSETS,ROOT

def load(mode='sample',tickers=None):
    tickers=tickers or ASSETS; folder=ROOT/'data'/('sample' if mode=='sample' else 'raw'); data={}
    for t in tickers:
        f=folder/f'{t.replace(".","_")}.csv'
        if f.exists():
            try: data[t]=pd.read_csv(f,index_col=0,parse_dates=True).dropna(how='all')
            except Exception: pass
    missing=[t for t in tickers if t not in data]
    if mode=='market' and missing:
        from src.data.downloader import download
        data.update(download(missing,cache=folder))
    if not data: raise ValueError(f'No usable {mode} data. Sample files should be included in data/sample.')
    return data
''',
'src/data/preprocessing.py':'''"""Chronological data preparation."""
import pandas as pd

def chronological_split(frame, fractions=(.70,.15,.15)):
    if abs(sum(fractions)-1)>1e-8: raise ValueError('fractions must sum to one')
    n=len(frame); a=int(n*fractions[0]); b=a+int(n*fractions[1])
    return frame.iloc[:a],frame.iloc[a:b],frame.iloc[b:]
''',
'src/features/technical_features.py':'''"""Point-in-time technical features and next-period labels."""
import numpy as np
import pandas as pd

def make_features(df):
    close=df['Close'].astype(float); ret=close.pct_change()
    vol=df.get('Volume',pd.Series(0,index=df.index)).astype(float)
    ma=close.rolling(20,min_periods=5).mean()
    delta=close.diff(); gain=delta.clip(lower=0).rolling(14,min_periods=5).mean(); loss=-delta.clip(upper=0).rolling(14,min_periods=5).mean()
    rs=gain/loss.replace(0,np.nan); rsi=100-100/(1+rs)
    x=pd.DataFrame({'daily_return':ret,'volatility':ret.rolling(20,min_periods=5).std(),'momentum':close.pct_change(10),'rsi':rsi,'moving_average':ma,'moving_average_ratio':close/ma-1,'rolling_mean_return':ret.rolling(10,min_periods=5).mean(),'volume_change':vol.pct_change().replace([np.inf,-np.inf],np.nan)},index=df.index)
    x['target']=(ret.shift(-1)>0).astype(int)
    return x.replace([np.inf,-np.inf],np.nan).dropna()
''',
}
for f,s in files.items():
 q=p/f;q.parent.mkdir(parents=True,exist_ok=True);q.write_text(s,encoding='utf-8')
for d in ['src','src/data','src/features','src/models','src/portfolio','src/utils','app','tests','notebooks','results','models','data/raw','data/processed','data/sample']:
 (p/d).mkdir(parents=True,exist_ok=True)
 (p/d/'__init__.py').touch() if d.startswith('src') else None
