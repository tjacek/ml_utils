import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import voting,filtr

def quality(**args):
    datasets=voting.get_datasets(**args)
    acc=[ pred_acc(data_i) for data_i in datasets]
    print(acc)
    return np.flip(np.argsort(acc))

def pred_acc(data_i):
    print(data_i.dim())
    train,test=filtr.split(data_i.info)
    train_data=filtr.filtered_dict(train,data_i)
    by_person=samples_by_person(train)
    def one_out(i):
        one,rest=by_person[i],[]
        for person_j,names_j in by_person.items():
            if(person_j!=i):
                rest+=names_j
        return one,rest
    acc=[]
    for person_i in by_person.keys():
        print(person_i)	
        one,rest=one_out(person_i)
        X_rest,y_rest=as_arrays(rest,train_data)
        clf_i=LogisticRegression()
        clf_i.fit(X_rest,y_rest)
        X_one,y_one=as_arrays(one,train_data)
        y_pred=clf_i.predict(X_one)
        acc.append(accuracy_score(y_one,y_pred))
    print(acc)
    return np.mean(acc)

def samples_by_person(train):
    persons_dict=get_person( train)
    person_ids= list(np.unique(persons_dict.values()))
    by_person={person_i:
                    [name_j for name_j,person_j in persons_dict.items()
                        if(person_j==person_i)]
                            for person_i in person_ids}
    return by_person

def get_person(names):
    return { name_i:name_i.split('_')[1]  for name_i in names}

def as_arrays(names,data_i):
    X=np.array([data_i[name_i] for name_i in names])
    y=[int(name_i.split('_')[0]) -1 for name_i in names]
    return X,y