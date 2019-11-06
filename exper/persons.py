import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import exper.voting,filtr,learn

def pred_by_person(train_i,clf_type="LR"):
    one_out_i=one_out_person(train_i)
    pairs=[]
    for one_j,rest_j in one_out_i:
        print(len(one_j))
        print(len(rest_j))
        clf_j=learn.get_cls(clf_type)
        clf_j.fit(rest_j.X,rest_j.get_labels())
        dist_j=clf_j.predict_proba(one_j.X)
        pairs+=pred_vectors(rest_j,one_j,clf_type)
    return pairs

def one_out_person(train_i):
    person_id=unique_persons(train_i)
    one_out_i=[]
    for person_j in person_id:
        def selector_j(name_k):
            return name_k.split('_')[1]==person_j
        one_j,rest_j=train_i.split(selector_j)
        one_out_i.append((one_j,rest_j))
    return one_out_i

def pred_vectors(train_j,test_j,clf_type="LR"):
    clf_j=learn.get_cls(clf_type)
    clf_j.fit(train_j.X,train_j.get_labels())
    dist_j=clf_j.predict_proba(test_j.X)
    return list(zip(test_j.info,dist_j))

def unique_persons(data_i):
    persons_dict=filtr.get_person(data_i.info)
    return list(set(persons_dict))