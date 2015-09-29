import dataset
import sklearn.cross_validation as cv
import sklearn.grid_search as gs
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

class OptimizedSVM(object):
    def __init__(self):
        rbf={'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],'C': [1, 10, 100, 1000]}
        linear={'kernel': ['linear'], 'C': [1, 10, 100, 1000]}
        self.params=[rbf,linear]
        self.SVC=SVC(C=1)
        
    def grid_search(self,X_train,y_train,metric='accuracy'):
        clf = gs.GridSearchCV(self.SVC,self.params, cv=5,scoring=metric)
        clf.fit(X_train,y_train)
        return clf

class OptimizedRandomForest(object):
    def __init__(self):
        params={}
        params['n_estimators']=[50,100,300,400,500] 
        #params['criterion']=['gini','entropy']
        self.params=[params]
        self.rf= RandomForestClassifier(n_estimators=10)
    
    def grid_search(self,X_train,y_train,metric='accuracy'):
        clf = gs.GridSearchCV(self.rf,self.params, cv=5,scoring=metric)
        clf.fit(X_train,y_train)
        return clf

def random_eval(dataset):
    X=dataset.X
    y=dataset.y
    X_train, X_test, y_train, y_test = cv.train_test_split(
                                       X, y, test_size=0.5, random_state=0)
    svm_opt=OptimizedSVM()
    #svm_opt=OptimizedRandomForest()
    clf=svm_opt.grid_search(X_train,y_train)
    
    eval_train(clf)
    eval_test(X_test,y_test,clf)

def determistic_eval(train_path,test_path):
    train=dataset.labeled_to_dataset(train_path)
    test=dataset.labeled_to_dataset(test_path)
    #svm_opt=OptimizedSVM()
    svm_opt=OptimizedRandomForest()
    clf=svm_opt.grid_search(train.X,train.y)  
    eval_train(clf)
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
    print(classification_report(y_true, y_pred))

if __name__ == "__main__":  
    path="/home/user/df/exp3/dataset.lb"
    #dataset=dataset.labeled_to_dataset(path)
    #random_eval(dataset)
    train_path="train.lb"
    test_path="test.lb"
    determistic_eval(train_path,test_path)
