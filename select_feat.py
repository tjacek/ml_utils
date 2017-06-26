import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn import linear_model
from sklearn.ensemble import ExtraTreesClassifier
import sklearn.decomposition
from sklearn.linear_model import LassoCV

def lasso_model(data):
    clf = LassoCV()
    sfm = SelectFromModel(clf, threshold=0.1)
    sfm.fit(data.X, data.y)
   
    new_X= sfm.transform(data.X)
    
    #clf = sklearn.decomposition.PCA(n_components=50)
    #new_X=clf.fit_transform(data.X)
    
    data.X=new_X
    data.dim=new_X.shape[1]
    print("New dim %d " % data.dim)
    print(new_X.shape)
    return data