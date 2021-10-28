import numpy as np
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from random import randrange
import seqs

def reduce_dim(in_path,out_path,n_feats=32):
    ts=seqs.read_seqs(in_path)
    train,test= ts.split()
    X,y=subsampled_frames(train)
    svc = SVC(kernel="linear", C=1)
    rfe = RFE(estimator=svc, n_features_to_select=n_feats, step=1)
    rfe.fit(X, y)
    print("OK")
    def helper(x_i):
        x=rfe.transform(x_i)
        print(x.shape)
        return x
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

def subsampled_frames(ts):
    X,y=[],[]
    for name_i,x_i in ts.items():
        x_j=x_i[randrange(x_i.shape[0])]
        y_i=name_i.get_cat()
        X.append(x_j)
        y.append(y_i)
    return np.array(X),y

in_path="../conv_frames/seqs" 
reduce_dim(in_path,"shape_32")