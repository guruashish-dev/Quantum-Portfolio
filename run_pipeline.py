"""Run complete reproducible Q-Portfolio research workflow."""
import argparse
import time
import numpy as np,pandas as pd
from src.data.loader import load
from src.features.technical_features import make_features
from src.data.preprocessing import chronological_split
from src.models.classical_model import train,metrics
from src.models.quantum_model import QuantumClassifier
from src.utils.config import FEATURES,QUANTUM_FEATURES,ROOT
def progress(stage,detail,percent,ticker='',epoch=0,total=0):
 print(f'QPROGRESS|{stage}|{detail}|{percent:.4f}|{ticker}|{epoch}|{total}',flush=True)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['sample','market'],default='sample');ap.add_argument('--epochs',type=int,default=20);args=ap.parse_args();results=ROOT/'results';results.mkdir(exist_ok=True);(ROOT/'models').mkdir(exist_ok=True)
 progress('loading',f'Loading {args.mode} OHLCV data',.03);data=load(args.mode);rows=[];fmap={};tests={};models={};signals={}
 tickers=list(data);total_assets=len(tickers)
 for asset_index,ticker in enumerate(tickers):
  start=.06 + asset_index/total_assets*.58
  progress('features',f'{ticker}: calculating technical features',start,ticker)
  f=make_features(data[ticker]);tr,va,te=chronological_split(f);fmap[ticker]=f;tests[ticker]=te
  progress('classical',f'{ticker}: fitting logistic baseline',start+.08,ticker)
  classical_started=time.perf_counter();cm=train(tr[FEATURES],tr.target)
  def on_epoch(epoch,total,loss,ticker=ticker,start=start):
   progress('quantum',f'{ticker}: variational circuit epoch {epoch}/{total}',start+.10+.25*epoch/total,ticker,epoch,total)
  quantum_started=time.perf_counter();qm=QuantumClassifier(epochs=args.epochs).fit(tr[QUANTUM_FEATURES],tr.target,progress_callback=on_epoch);models[ticker]=(cm,qm);np.save(ROOT/'models'/f'{ticker}_quantum_parameters.npy',np.asarray(qm.weights))
  progress('scoring',f'{ticker}: evaluating held-out predictions',start+.40,ticker)
  for name,m,cols in [('classical',cm,FEATURES),('quantum',qm,QUANTUM_FEATURES)]:
    started=classical_started if name=='classical' else quantum_started
    pr=m.predict(te[cols]);pb=m.predict_proba(te[cols])[:,1];rows.append({'asset':ticker,'model':name,'runtime_seconds':time.perf_counter()-started,**metrics(te.target,pr,pb)})
    for date,prob in zip(te.index,pb):signals.setdefault(date,{})[(ticker,'qml' if name=='quantum' else 'classical')]=float(prob)
 progress('scoring','Writing model comparison',.69);pd.DataFrame(rows).to_csv(results/'model_comparison.csv',index=False)
 # Most recent available, point-in-time predictions and constrained portfolio.
 progress('allocation','Building point-in-time portfolio signals',.73);alloc=[]
 for ticker,df in data.items():
  cm,qm=models[ticker];latest=fmap[ticker].iloc[-1];prob=float(qm.predict_proba([latest[QUANTUM_FEATURES]])[0,1]);alloc.append({'asset':ticker,'predicted_direction':int(prob>=.5),'prediction_probability':prob,'expected_return':(prob-.5)*.04})
 from src.portfolio.optimizer import optimize
 from src.portfolio.backtester import backtest
 prices=pd.concat([d.Close.rename(t) for t,d in data.items()],axis=1).dropna();mu=np.array([x['expected_return'] for x in alloc]);cov=prices.pct_change().dropna().cov().values;weights=optimize(mu,cov)
 for row,w in zip(alloc,weights):row['portfolio_weight']=float(w)
 pd.DataFrame(alloc).to_csv(results/'portfolio_allocation.csv',index=False)
 progress('backtest','Running walk-forward backtest',.84);sig=pd.DataFrame.from_dict(signals,orient='index').sort_index();bt,summary=backtest(prices,sig);bt.to_csv(results/'backtest_results.csv',index=False)
 summary_frame=pd.DataFrame(summary).T
 runtime_summary=pd.DataFrame(rows).groupby('model')['runtime_seconds'].sum()
 summary_frame['runtime_seconds']=[runtime_summary.get({'qml':'quantum'}.get(index,index),0.) for index in summary_frame.index]
 summary_frame=summary_frame.rename(index={'qml':'quantum'})
 summary_frame.to_csv(results/'backtest_summary.csv')
 comparison=summary_frame.loc[[name for name in ['classical','quantum'] if name in summary_frame.index], ['expected_return','volatility','sharpe_ratio','max_drawdown','runtime_seconds']].reset_index(names='model')
 if {'classical','quantum'}.issubset(set(comparison['model'])):
  delta=comparison.set_index('model').loc['quantum']-comparison.set_index('model').loc['classical']
  comparison=pd.concat([comparison,pd.DataFrame([{'model':'quantum_minus_classical',**delta.to_dict()}])],ignore_index=True)
 comparison.to_csv(results/'performance_comparison.csv',index=False)
 import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
 if not bt.empty:
  for col,title in [('portfolio_value','portfolio_value.png'),('drawdown','drawdown.png')]:
   if col=='drawdown':
    bt['drawdown']=bt.groupby('strategy').portfolio_value.transform(lambda s:s/s.cummax()-1)
   fig,ax=plt.subplots();
   for strategy,g in bt.groupby('strategy'):ax.plot(g.date,g[col],label=strategy)
   ax.legend();ax.set_title(col);fig.tight_layout();fig.savefig(results/title);plt.close(fig)
 progress('complete','Analysis complete',1.0);print('Model metrics\n',pd.DataFrame(rows).to_string(index=False));print('Portfolio\n',pd.DataFrame(alloc).to_string(index=False));print('Backtest\n',pd.DataFrame(summary).T.to_string())
if __name__=='__main__':main()


