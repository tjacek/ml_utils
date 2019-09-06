import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import voting,filtr,learn

def in_sample(data_i,clf_type="LR"):
    train,test=filtr.split(data_i.info)
    train_data=filtr.filtered_dict(train,data_i)
    by_person=samples_by_person(train)
    person_pred=pred_by_person(train_data,by_person,clf_type)
    pairs=[]
    for person_i,pred_i in person_pred.items():
        one,y_one,y_pred=pred_i
        pairs+=zip(one,y_pred)
    return pairs

def samples_by_person(train):
    persons_dict=get_person( train)
    person_ids= list(np.unique(persons_dict.values()))
    by_person={person_i:
                    [name_j for name_j,person_j in persons_dict.items()
                        if(person_j==person_i)]
                            for person_i in person_ids}
    return by_person

def pred_by_person(train_data,by_person,clf_type):
    def one_out(i):
        one,rest=by_person[i],[]
        for person_j,names_j in by_person.items():
            if(person_j!=i):
                rest+=names_j
        return one,rest
    def pred_helper(person_i):
        one,rest=one_out(person_i)
        X_rest,y_rest=as_arrays(rest,train_data)
        clf_i=learn.get_cls(clf_type)
        clf_i.fit(X_rest,y_rest)
        X_one,y_one=as_arrays(one,train_data)
        y_pred=clf_i.predict(X_one)
        return (one,y_one,y_pred)
    acc=[]
    return { person_i:pred_helper(person_i) for person_i in by_person.keys()}

def get_person(names):
    return { name_i:name_i.split('_')[1]  for name_i in names}

def as_arrays(names,data_i):
    X=np.array([data_i[name_i] for name_i in names])
    y=[int(name_i.split('_')[0]) -1 for name_i in names]
    return X,y