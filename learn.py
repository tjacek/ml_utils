import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_recall_fscore_support,accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import plot

def get_cls(clf_type):
    if(clf_type=="SVC"):
        print("SVC")
        return make_SVC()
    elif(clf_type=="MLP"):
        print("MLP")
        return make_mlp()
    else:
        print("LR")
        return LogisticRegression(solver='liblinear')#max_iter=1000)

def make_SVC():
    params=[{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],'C': [1, 10, 50,110, 1000]},
            {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]    
    clf = GridSearchCV(SVC(C=1,probability=True),params, cv=5,scoring='accuracy')
    return clf

def make_mlp():
    return MLPClassifier(alpha=1, max_iter=1000)

def show_result(y_pred,y_true=None,names=None):
    if((not y_true) or (not names)):
        y_pred,y_true,names=y_pred
    print(classification_report(y_true, y_pred,digits=4))
    print(show_errors(y_pred,y_true,names))
    
def show_confusion(result,out_path=None):
    cf_matrix=confusion_matrix(result[0],result[1])
    cf_matrix=pd.DataFrame(cf_matrix,index=range(cf_matrix.shape[0]))
    if(out_path):
        np.savetxt(out_path,cf_matrix,delimiter=",",fmt='%.2e')
    else:
        labels=["predicted labels","true labels"]
        plot.heat_map(cf_matrix,labels)

def show_errors(y_pred,y_true,names):
    errors= [ pred_i!=true_i 
            for pred_i,true_i in zip(y_pred,y_true)]
    error_names=[ (names[i],y_pred[i])
                  for i,error_i in enumerate(errors)
                    if(error_i)]
    return error_names

def compute_score(y_true,y_pred,as_str=True):
    precision,recall,f1,support=precision_recall_fscore_support(y_true,y_pred,average='weighted')
    accuracy=accuracy_score(y_true,y_pred)
    if(as_str):
        return "%0.4f,%0.4f,%0.4f,%0.4f" % (accuracy,precision,recall,f1)
    else:
        return (accuracy,precision,recall,f1)

def acc_arr(results):
    return [accuracy_score(result_i[0],result_i[1]) 
                for result_i in results]