import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn import linear_model
from sklearn.ensemble import ExtraTreesClassifier
import sklearn.decomposition
from sklearn.linear_model import LassoCV

def select_feat(data,method='pca'):
    if(method=='lasso'):
        new_X=lasso_select(data)
    else:
        new_X=pca_select(data)
    return data.new_dataset(new_X)

def lasso_select(data):
    clf = LassoCV()
    sfm = SelectFromModel(clf, threshold=0.1)
    sfm.fit(data.X, data.y)
    new_X= sfm.transform(data.X)
    return new_X

def pca_select(data):
    clf = sklearn.decomposition.PCA(n_components=50)
    new_X=clf.fit_transform(data.X)
    return new_X