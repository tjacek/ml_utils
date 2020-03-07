import numpy as np
from sklearn.metrics import accuracy_score
import feats,learn

def orth_selection(in_path):
    votes=feats.read_list(in_path)
    train=[ vote_i.split()[0] for vote_i in votes]
    acc_i=[get_acc(train_i) for train_i in train]
    acc_i=np.argsort(acc_i)
    print(acc_i)

def get_acc(train_i):
    person_i=[ int(info_j.split("_")[1])
                for info_j in train_i.info]
    clf_i=learn.get_cls("LR")
    clf_i.fit(train_i.X,person_i)
    person_predict=clf_i.predict(train_i.X)
    return accuracy_score(person_i,person_predict)