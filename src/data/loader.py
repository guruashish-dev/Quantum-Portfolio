"""Load sample or cached/downloaded market OHLCV."""
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
