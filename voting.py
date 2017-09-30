import sklearn
print(sklearn.__version__)
from sklearn import datasets
import sklearn.cross_validation as cv
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import VotingClassifier
import exper
from sklearn.svm import SVC
import pandas as pd
import eval
from collections import Counter

class Ensemble(object):
    def __init__(self,optim_cls):
        self.optim_cls=optim_cls

    def __call__(self,datasets):
        preds=[ self.optim_cls(data_i)
                    for data_i in datasets]
        true_y=self.optim_cls.true_y
        sample_pred=get_sample_pred(preds)
        ensemble_pred=elect(sample_pred)
        print(self.stats(sample_pred,ensemble_pred))
        return ensemble_pred,true_y

    def stats(self,sample_pred,ensemble_pred):
        n=len(sample_pred[0])
        def count_helper(pred_j,votes):
            win_votes=[int(pred_j==vote_i) 
                        for vote_i in votes]
            return sum(win_votes)
        return [ count_helper(pred_j,votes_j)
                    #for votes_j in sample_pred]
                    for votes_j,pred_j in zip(sample_pred,ensemble_pred)]


class SimpleCls(object):
    def __init__(self,simple_cls):
        self.simple_cls=simple_cls
        self.true_y=None

    def __call__(self,data):
        even_data,odd_data=exper.split_data(data)
        optim=self.simple_cls()
        try:
            clf = optim.grid_search(odd_data.X, odd_data.y)
        except AttributeError:
            clf = optim
        
        clf = clf.fit(odd_data.X, odd_data.y)
        self.true_y=even_data.y
        pred_y= clf.predict(even_data.X)
        return pred_y

def get_ensemble(cls_type='svm'):
    if(cls_type=='rf'):
        basic_cls=eval.OptimizedRandomForest     
    elif(cls_type=='lr'):
        basic_cls=LogisticRegression
    else:    
        basic_cls=eval.OptimizedSVM
    optim_cls=SimpleCls(basic_cls)
    return Ensemble(optim_cls)

def get_sample_pred(preds):
    def helper(i):
        return [  pred_j[i]
                  for pred_j in preds]
    n_samlpes=len(preds[0])
    return [ helper(i)
             for i in range(n_samlpes)]

def elect(preds):   
    def elect_helper(votes):
    	count =Counter(votes)
        elect_pair=count.most_common()[0]
        n_votes=elect_pair[0]
        return n_votes
    final_pred=[ elect_helper(votes_i) 
                 for votes_i in preds]
    return final_pred

def show_result(y_pred,y_true,conf=True):
    print(classification_report(y_true, y_pred,digits=4))
    if(conf):
       show_confusion(confusion_matrix(y_true, y_pred))
    
def show_confusion(cf_matrix):
    cf_matrix=pd.DataFrame(cf_matrix,index=range(cf_matrix.shape[0]))
    print(cf_matrix)