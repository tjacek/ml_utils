import numpy as np
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
import seqs

def reduce_dim(in_path,out_path):
    ts=seqs.read_seqs(in_path)
    X,y=as_frames(ts)
    svc = SVC(kernel="linear", C=1)
    rfe = RFE(estimator=svc, n_features_to_select=120, step=1)
    rfe.fit(X, y)
    def helper(x_i):
        return rfe.predict(x_i)
    ts.transform(helper)
    ts.save(out_path)

def as_frames(ts):
    X,y=[],[]
    for name_i,x_i in ts.items():
        y_i=name_i.get_cat()
        for x_j in x_i:
            X.append(x_j)
            y.append(y_i)
    return np.array(X),y


in_path="../deep_dtw/seqs" 
reduce_dim(in_path,"shape/120")