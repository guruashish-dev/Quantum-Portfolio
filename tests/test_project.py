import numpy as np,pandas as pd
from src.features.technical_features import make_features
from src.data.preprocessing import chronological_split
from src.portfolio.optimizer import optimize
from src.portfolio.risk import risk_metrics
def test_features_and_chronological_split():
 i=pd.bdate_range('2020-01-01',periods=150);c=np.linspace(100,120,150)+np.sin(np.arange(150));f=make_features(pd.DataFrame({'Close':c,'Volume':np.arange(150)+100},index=i));a,b,d=chronological_split(f);assert a.index.max()<b.index.min()<d.index.min();assert f.target.isin([0,1]).all()
def test_optimizer_constraints():
 w=optimize([.02,.01,.03],np.eye(3)*.01,max_weight=.4);assert np.isclose(w.sum(),1);assert ((w>=0)&(w<=.4+1e-8)).all()
def test_risk():
 x=risk_metrics([100,101,99,103]);assert x['max_drawdown']<0 and np.isfinite(x['sharpe_ratio'])
def test_quantum_exec_prediction():
 from src.models.quantum_model import QuantumClassifier
 X=np.random.default_rng(2).normal(size=(12,4));y=np.array([0,1]*6);m=QuantumClassifier(epochs=1).fit(X,y);p=m.predict_proba(X[:3]);assert p.shape==(3,2);assert len(m.loss_history)==1
