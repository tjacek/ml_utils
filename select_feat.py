import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn import linear_model
from sklearn.ensemble import ExtraTreesClassifier
import sklearn.decomposition
from sklearn.linear_model import LassoCV
from sklearn.svm import SVC
from sklearn.feature_selection import RFE


def select_feat(data,method='pca'):
    if(method==None):
        return data
    if(type(method)==tuple):
        method,n_feats=method
        print("selection " + method)
        if(method=='rfe'):
            new_X= rfe_select(data,n_feats)
        else:
            new_X=pca_select(data,n_feats)
    else:
        print("selection lasso" )
        new_X=lasso_select(data)
    return data.new_dataset(new_X)

def lasso_select(data):
    clf = LassoCV(max_iter=100)
    sfm = SelectFromModel(clf,threshold=0.01)
    sfm.fit(data.X, data.y)
    new_X= sfm.transform(data.X)
    print("New dim: ")
    print(new_X.shape)
    return new_X

def pca_select(data,n_select):
    clf = sklearn.decomposition.PCA(n_components=n_select)
    new_X=clf.fit_transform(data.X)
    return new_X

def rfe_select(data,n):
    svc = SVC(kernel='linear',C=1)
    rfe = RFE(estimator=svc,n_features_to_select=n,step=1)
    rfe.fit(data.X, data.y)
    new_X= rfe.transform(data.X)
    print("New dim: ")
    print(new_X.shape)
    return new_X
