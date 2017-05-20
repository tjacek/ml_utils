import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn import linear_model
from sklearn.ensemble import ExtraTreesClassifier

def lasso_model(data):
    #clf = ExtraTreesClassifier()
    clf=linear_model.Lasso(alpha=0.4)
    clf.fit(data.X,data.y)
    model = SelectFromModel(clf, prefit=True)
    new_X= model.transform(data.X)
    data.X=new_X
    data.dim=new_X.shape[1]
    print("New dim %d " % data.dim)
    print(new_X.shape)
    return data