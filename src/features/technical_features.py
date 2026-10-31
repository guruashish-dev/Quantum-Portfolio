"""Point-in-time technical features and next-period labels."""
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
