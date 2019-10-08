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
        pairs+=pred_vectors(rest_j,one_j,clf_type)#list(zip(one_j.info,dist_j))
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

#def in_sample(data_i,clf_type="LR"):
#    train,test=filtr.split(data_i.info)
#    train_data=filtr.filtered_dict(train,data_i)
#    by_person=samples_by_person(train)
#    person_pred=pred_by_person(train_data,by_person,clf_type)
#    pairs=[]
#    for person_i,pred_i in person_pred.items():
#        one,y_one,y_pred=pred_i
#        pairs+=zip(one,y_pred)
#    return pairs

#def samples_by_person(train):
#    persons_dict=get_person( train)
#    person_ids=  set(persons_dict.values())#list(np.unique(persons_dict.values()))
#    by_person={person_i:
#                    [name_j for name_j,person_j in persons_dict.items()
#                        if(person_j==person_i)]
#                            for person_i in person_ids}
#    return by_person

#def pred_by_person(train_data,by_person,clf_type):
#    def one_out(i):
#        one,rest=by_person[i],[]
#        for person_j,names_j in by_person.items():
#            if(int(person_j)!=i):
#                rest+=names_j
#        return one,rest
#    def pred_helper(person_i):
#        one,rest=one_out(person_i)
#        X_rest,y_rest=as_arrays(rest,train_data)
#        clf_i=learn.get_cls(clf_type)
#        clf_i.fit(X_rest,y_rest)
#        X_one,y_one=as_arrays(one,train_data)
#        y_pred=clf_i.predict(X_one)
#        return (one,y_one,y_pred)
#    acc=[]
#    return { person_i:pred_helper(person_i) for person_i in by_person.keys()}

def unique_persons(data_i):
    persons_dict=get_person(data_i.info)
    return list(set(persons_dict.values()))

def get_person(names):
    return { name_i:name_i.split('_')[1]  for name_i in names}

#def as_arrays(names,data_i):
#    X=np.array([data_i[name_i] for name_i in names])
#    y=[int(name_i.split('_')[0]) -1 for name_i in names]
#    return X,y