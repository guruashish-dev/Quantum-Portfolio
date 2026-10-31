"""Trainable four-wire PennyLane variational binary classifier."""
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
class QuantumClassifier:
 def __init__(self,layers=2,backend="auto",epochs=20,seed=42,stepsize=.08):
  self.layers=layers;self.epochs=epochs;self.seed=seed;self.backend=backend;self.loss_history=[]
  candidates=["lightning.qubit","default.qubit"] if backend=="auto" else [backend]
  for name in candidates:
   try:self.dev=qml.device(name,wires=4);self.backend=name;break
   except Exception:continue
  else:raise RuntimeError(f"Could not initialize PennyLane backend: {candidates}")
  layers_=layers
  @qml.qnode(self.dev,interface="autograd")
  def circuit(x,w):
   for i in range(4):qml.RY(x[i],wires=i)
   for k in range(layers_):
    for i in range(4):qml.RY(w[k,i,0],wires=i);qml.RZ(w[k,i,1],wires=i)
    for i in range(4):qml.CNOT(wires=[i,(i+1)%4])
   return qml.expval(qml.PauliZ(0))
  self.circuit=circuit;self.weights=pnp.array(np.random.default_rng(seed).normal(0,.1,(layers,4,2)),requires_grad=True);self.stepsize=stepsize
 def _scale(self,X):return np.pi*np.tanh(np.asarray(X,dtype=float))
 def fit(self,X,y,progress_callback=None):
  X=pnp.array(self._scale(X));y=pnp.array(np.asarray(y,dtype=float)*2-1);
  if len(X)>128:
   ix=np.random.default_rng(self.seed).choice(len(X),128,replace=False);X=X[ix];y=y[ix]
  opt=qml.AdamOptimizer(self.stepsize)
  def loss(w):return pnp.mean(pnp.stack([(self.circuit(x,w)-t)**2 for x,t in zip(X,y)]))
  for epoch in range(self.epochs):
   self.weights,l=opt.step_and_cost(loss,self.weights);self.loss_history.append(float(l))
   if progress_callback:
    progress_callback(epoch + 1, self.epochs, float(l))
  return self
 def predict_proba(self,X):
  z=np.array([float(self.circuit(pnp.array(x),self.weights)) for x in self._scale(X)]);p=np.clip((1-z)/2,0,1);return np.c_[1-p,p]
 def predict(self,X):return (self.predict_proba(X)[:,1]>=.5).astype(int)
 def circuit_text(self):return qml.draw(self.circuit)(pnp.zeros(4),self.weights)
if __name__=="__main__":
 from src.data.loader import load
 from src.features.technical_features import make_features
 from src.utils.config import QUANTUM_FEATURES
 f=make_features(next(iter(load().values())));m=QuantumClassifier().fit(f[QUANTUM_FEATURES],f.target);print(m.loss_history)

