"""Out-of-sample walk-forward allocation using only prior return covariance."""
import numpy as np,pandas as pd
from src.portfolio.optimizer import optimize
from src.portfolio.risk import risk_metrics
def backtest(price_frame,signals,initial_capital=100000,max_weight=.4,risk_aversion=5):
 returns=price_frame.pct_change().dropna(); rows=[]; values={k:float(initial_capital) for k in ['qml','classical','equal_weight','buy_hold']}
 for date,r in returns.iterrows():
  if date not in signals.index:continue
  hist=returns.loc[:date].iloc[:-1].tail(60)
  if len(hist)<10:continue
  out={}
  for model in ['qml','classical']:
   probs=[]
   for asset in price_frame.columns:
    key=(asset,model)
    probs.append(float(signals.loc[date,key]) if key in signals.columns and pd.notna(signals.loc[date,key]) else .5)
   w=optimize((np.asarray(probs)-.5)*.04,hist.cov().fillna(0).values,risk_aversion,max_weight);out[model]=float(w@r.reindex(price_frame.columns).values)
  out['equal_weight']=float(r.mean());out['buy_hold']=float(r.mean())
  for model,dr in out.items():
   values[model]*=1+dr;rows.append({'date':date,'strategy':model,'portfolio_value':values[model],'daily_return':dr})
 frame=pd.DataFrame(rows);return frame,{k:risk_metrics(g.portfolio_value.to_numpy()) for k,g in frame.groupby('strategy')}
