import re
from collections import defaultdict
import random,copy
import feats

def filtered_dict(names,dic):
    if(type(dic)==feats.FeatureSet):
        dic=dic.to_dict()
    return { name_i:dic[name_i] for name_i in names}

def train_by_cat(ts_dataset):
    names=ts_dataset.ts_names()
    train,test=split(names,get_person)
    return by_cat(train)

def by_cat(names):
#    names=clean_str(names)
    names_by_cat=defaultdict(lambda:[])
    for name_i in names:
    	names_by_cat[get_cat(name_i)].append(name_i)
    return names_by_cat

def random_pairs(names):
    random_names=copy.copy(names)
    random.shuffle(random_names)
    return [ (x_i,y_i) for x_i,y_i in zip(names,random_names)]

def split(names,selector=None):
    if(not selector):
        selector=get_person
    train,test=[],[]
    for name_i in names:
        if(selector(name_i)):
            train.append(name_i)
        else:
            test.append(name_i)
    return train,test

def get_cat(name_i):
    return int(name_i.split('_')[0])	

def get_person(name_i):
    return (int(name_i.split('_')[1])%2)==1

def clean_str(names):
    return [re.sub(r'[a-z]','',str_i.strip()) 
                for str_i in names]