import select_feat

import pandas as pd
import numpy as np
import dataset
import sklearn.cross_validation as cv
import sklearn.grid_search as gs
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import PredefinedSplit
from sklearn.cross_validation import LeaveOneOut

class OptimizedSVM(object):
    def __init__(self):
        rbf={'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],'C': [1, 10, 50,110, 1000]}
        linear={'kernel': ['linear'], 'C': [1, 10, 100, 1000]}
        self.params=[rbf,linear]
        self.SVC=SVC(C=1)
        
    def grid_search(self,X_train,y_train,n_split=5,metric='accuracy'):
        #validation_search(self.params)
        clf = gs.GridSearchCV(self.SVC,self.params, cv=n_split,scoring=metric)
        clf.fit(X_train,y_train)
        return clf

    def predefined_search(self,X_train,y_train,metric='accuracy'):
        n=len(y_train)
        self.params
        split=LeaveOneOut(n)
        clf = gs.GridSearchCV(self.SVC,self.params, cv=split,scoring=metric)
        clf.fit(X_train,y_train)
        print(clf.best_params_)
        return clf

class OptimizedRandomForest(object):
    def __init__(self):
        params={}
        params['n_estimators']=[50,100,300,400,500,700 ,1000] 
        #params['criterion']=['gini','entropy']
        self.params=[params]
        self.rf= RandomForestClassifier(n_estimators=10)
    
    def grid_search(self,X_train,y_train,n_split=5,metric='accuracy'):
        clf = gs.GridSearchCV(self.rf,self.params, cv=n_split,scoring=metric)
        clf.fit(X_train,y_train)
        return clf

    def predefined_search(self,X_train,y_train,metric='accuracy'):
        n=len(y_train)
        split=LeaveOneOut(n)
        clf = gs.GridSearchCV(self.rf,self.params, cv=split,scoring=metric)
        print(clf)
        clf.fit(X_train,y_train)
        print(clf.best_params_)
        return clf

def eval_test(test,clf):
    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print(type(test))
    X_test=test.X
    y_test=test.y
    y_true, y_pred = y_test, clf.predict(X_test)
    result=(y_true==y_pred)
    show_confusion(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred,digits=4))
    
def show_confusion(cf_matrix):
    cf_matrix=pd.DataFrame(cf_matrix,index=range(cf_matrix.shape[0]))
    print(cf_matrix)

def determistic_eval(train,test,svm=False):
    if(svm):
        clf=OptimizedSVM()
    else:
        clf=OptimizedRandomForest()
    clf = clf.grid_search(train.X, train.y)#RandomForestClassifier(n_estimators=1000)
    print(train.y)
    clf = clf.fit(train.X, train.y)
    eval_test(test,clf)

if __name__ == "__main__":
    in_path='../reps/dtw_feat/dataset.txt'
    in_path2= '../ultimate3/simple/dataset.txt'#'../reps/dtw_feat/simple3/dataset.txt'
    data=dataset.read_and_unify(in_path,in_path2)
    #data=dataset.get_annotated_dataset(in_path2)
    #data=select_feat.lasso_model(data)
    even_data=dataset.select_person(data,i=0)
    odd_data=dataset.select_person(data,i=1)
    #even_data,odd_data = dataset.select_single(data,i=4)
    print(len(even_data))
    print(len(odd_data))
    determistic_eval(odd_data,even_data,svm=False)
    #random_eval(dataset,svm=False)