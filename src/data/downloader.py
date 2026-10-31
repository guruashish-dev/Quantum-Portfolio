"""Cached yfinance OHLCV download helper."""
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
