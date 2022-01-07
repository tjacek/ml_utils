import sys
sys.path.append("..")
sys.path.append("../optim")
from sklearn.metrics import roc_auc_score
import pref

class AcuracyLoss(object):
    def __init__(self,train_dict):
        self.train_dict=train_dict
        self.n_calls=0

    def __call__(self,score):
        result=pref.eval_score(score,self.train_dict)
        acc=result.get_acc()
        print(acc)
        self.n_calls+=1
        return (1.0-acc)

class AucLoss(object):
    def __init__(self,train_dict):
        self.train_dict=train_dict
        self.n_calls=0

    def __call__(self,score):
        result=pref.eval_score(score,self.train_dict)
        y_true,y_pred= result.as_labels()
        n_cats=result.n_cats()
        y_pred=learn.to_one_hot(y_pred,n_cats)
        y_true=learn.to_one_hot(y_true,n_cats)
        auc_ovo = roc_auc_score(y_true,y_pred,multi_class="ovo")
#        print(auc_ovo)
        self.n_calls+=1
        return -1.0*auc_ovo