import numpy as np
import re
from collections import defaultdict
import random,copy

def filtered_dict(names,dic):
    return { name_i:dic[name_i] for name_i in names}

def train_by_cat(ts_dataset):
    names=ts_dataset.ts_names()
    train,test=split(names,get_person)
    return by_cat(train)

def by_cat(names):
    names_by_cat=defaultdict(lambda:[])
    for name_i in names:
    	names_by_cat[get_cat(name_i)].append(name_i)
    return names_by_cat

def by_person(names):
    persons=all_persons(names)
    persons_id=np.unique(persons)
    def person_helper(i):
        selector=lambda s_i:int(s_i.split('_')[1])==i
        return split(names,selector)[0]
    return [person_helper(i) for i in persons_id]    

def random_pairs(names):
    random_names=copy.copy(names)
    random.shuffle(random_names)
    return [ (x_i,y_i) for x_i,y_i in zip(names,random_names)]

def split(names,selector=None):
    if(type(names)==dict):
        train,test=split(names.keys(),selector)
        return filtered_dict(train,names),filtered_dict(test,names)
    if(not selector):
        selector=get_person
    train,test=[],[]
    for name_i in names:
        if(selector(name_i)):
            train.append(name_i)
        else:
            test.append(name_i)
    return train,test

def one_per_person(names):
    alredy_in=set([])
    per_person=[]
    for name_i in names:
        id_i="_".join(name_i.split('_')[:2])
        if(not (id_i in alredy_in)):
            alredy_in.add(id_i)
            per_person.append(name_i)
    return per_person

def n_cats(names):
    return np.unique(all_cats(names)).shape[0]

def all_cats(names):
    return [ get_cat(name_i) for name_i in names]

def get_cat(name_i):
    return int(name_i.split('_')[0])-1

def all_persons(names):
    return [ int(name_i.split('_')[1]) for name_i in names]

def get_person(name_i):
    return (int(name_i.split('_')[1])%2)==1

def clean_str(names):
    return [re.sub(r'[a-z]','',str_i.strip()) 
                for str_i in names]

def erros(names,y_true,y_pred):
    err=[true_i!=pred_i for true_i,pred_i in zip(y_true,y_pred)]
    return [name_i for i,name_i in enumerate(names)
                if(err[i])]

def cats_id(names):
    return np.unique(all_cats(names))

def ordered_cats(pairs_dict):
    uniqe_cats=cats_id(pairs_dict.keys())
    uniqe_cats={ cat_i:i for i,cat_i in enumerate(uniqe_cats)} 
    new_pairs={}
    for name_i,value_i in pairs_dict.items():
        raw=name_i.split('_')
        raw[0]= str(uniqe_cats[int(raw[0])-1]+1)
        new_name="_".join(raw)
        new_pairs[new_name]=value_i
    return new_pairs