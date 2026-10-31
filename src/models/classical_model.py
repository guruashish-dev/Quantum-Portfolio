"""Logistic regression baseline."""
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix
from src.utils.config import ROOT,FEATURES
def metrics(y,p,prob):
 d={"accuracy":accuracy_score(y,p),"precision":precision_score(y,p,zero_division=0),"recall":recall_score(y,p,zero_division=0),"f1":f1_score(y,p,zero_division=0),"confusion_matrix":confusion_matrix(y,p).tolist()}
 try:d["roc_auc"]=roc_auc_score(y,prob)
 except ValueError:d["roc_auc"]=float("nan")
 return d
def train(x,y,save=True):
 m=make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),LogisticRegression(random_state=42,max_iter=1000));m.fit(x,y)
 if save: (ROOT/"models").mkdir(exist_ok=True);joblib.dump(m,ROOT/"models/classical.joblib")
 return m
if __name__=="__main__":
 from src.data.loader import load
 from src.features.technical_features import make_features
 from src.data.preprocessing import chronological_split
 d=next(iter(load().values())); f=make_features(d); a,_,c=chronological_split(f);m=train(a[FEATURES],a.target);print(metrics(c.target,m.predict(c[FEATURES]),m.predict_proba(c[FEATURES])[:,1]))
