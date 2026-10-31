"""Long-only mean variance portfolio optimizer."""
import numpy as np
from scipy.optimize import minimize
def optimize(expected_returns,covariance,risk_aversion=5,max_weight=.4):
 mu=np.asarray(expected_returns,float);cov=np.asarray(covariance,float);n=len(mu)
 if n*max_weight<1:raise ValueError("max_weight too small to fully invest")
 def obj(w):return -(w@mu-risk_aversion*(w@cov@w))
 r=minimize(obj,np.ones(n)/n,bounds=[(0,max_weight)]*n,constraints=[{"type":"eq","fun":lambda w:w.sum()-1}],method="SLSQP",options={"maxiter":500,"ftol":1e-12})
 if not r.success:raise ValueError(r.message)
 return r.x
