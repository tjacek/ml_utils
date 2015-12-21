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
import sys

class OptimizedSVM(object):
    def __init__(self):
        rbf={'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],'C': [1, 10, 50,110, 1000]}
        linear={'kernel': ['linear'], 'C': [1, 10, 100, 1000]}
        self.params=[rbf,linear]
        self.SVC=SVC(C=1)
        
    def grid_search(self,X_train,y_train,n_split=5,metric='accuracy'):
        validation_search(self.params)
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
        params['n_estimators']=[50,100,300,400,500,700] 
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

def  unify_dataset(X_train,y_train,X_test,y_test):
     split_train=[-1 for i in range(len(y_train))]
     split_test=[0 for i in range(len(y_test))]
     split=split_train+split_test
     X = np.concatenate((X_train, X_test))
     y = np.concatenate((y_train, y_test))
     ps=PredefinedSplit(split)
     print(type(ps))
     return X,y,ps

def random_eval(dataset,svm=False):
    X=dataset.X
    y=dataset.y
    X_train, X_test, y_train, y_test = cv.train_test_split(
                                       X, y, test_size=0.5, random_state=0)
    if(svm):
        svm_opt=OptimizedSVM()
    else:
       svm_opt=OptimizedRandomForest()
    clf=svm_opt.grid_search(X_train,y_train)
    
    eval_train(clf)
    eval_test(X_test,y_test,clf)

def determistic_eval(train_path,test_path,svm=False):
    train=dataset.labeled_to_dataset(train_path)
    test=dataset.labeled_to_dataset(test_path)
    if(svm):
        svm_opt=OptimizedSVM()
    else:
        svm_opt=OptimizedRandomForest()
    clf = RandomForestClassifier(n_estimators=400)
    #clf=svm_opt.grid_search(train.X,train.y,n_split=2)  
    #clf=svm_opt.predefined_search(train.X,train.y)  
    #eval_train(clf)
    print(train.y)
    clf = clf.fit(train.X, train.y)
    eval_test(test.X,test.y,clf)

def eval_train(clf):
    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() * 2, params))
    print()

def eval_test(X_test,y_test,clf):
    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    show_confusion(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred))
    
def show_confusion(cf_matrix):
    cf_matrix=pd.DataFrame(cf_matrix,index=range(cf_matrix.shape[0]))
    print(cf_matrix)


def validation_search(params):
    print(params)
    #for key in params.keys():
    #    print(key)

if __name__ == "__main__":
    path="/home/user/cf/seqs/"
    if(len(sys.argv)>1):
        random=int(sys.argv[1])
        random=bool(random)
    else:
        random=False#True
    if(random):
        in_path="../af/cascade/full_dataset"#"../af/result/full_dataset"
        dataset=dataset.labeled_to_dataset(in_path)
        random_eval(dataset)
    else:
        train_path="../af/cascade/full_dataset_train"
        test_path="../af/cascade/full_dataset_test"
        determistic_eval(train_path,test_path,False)
