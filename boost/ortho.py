import numpy as np
from sklearn.metrics import accuracy_score
import feats,learn, exper.selection


def get_acc(path_i):
    clf_ord=orth_selection(path_i)
    results=exper.selection.selected_voting(path_i,clf_ord)
    return [accuracy_score(result_i[0],result_i[1]) 
                for result_i in results]

def orth_selection(in_path):
    votes=feats.read_list(in_path)
    train=[ vote_i.split()[0] for vote_i in votes]
    acc_i=[person_acc(train_i) for train_i in train]
    clf_ord=np.flip(np.argsort(acc_i))
    return clf_ord

def person_acc(train_i):
    person_i=[ int(info_j.split("_")[1])
                for info_j in train_i.info]
    clf_i=learn.get_cls("LR")
    clf_i.fit(train_i.X,person_i)
    person_predict=clf_i.predict(train_i.X)
    return accuracy_score(person_i,person_predict)