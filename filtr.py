import re
from collections import defaultdict

def train_by_cat(ts_dataset):
    names=clean_str(ts_dataset.ts_names())
    train,test=split(names,get_person)
    return by_cat(train)

def by_cat(names):
    names_by_cat=defaultdict(lambda:[])
    for name_i in names:
    	names_by_cat[get_cat(name_i)].append(name_i)
    return names_by_cat

def split(names,selector):
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