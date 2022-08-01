import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from collections import defaultdict
import feats,dtw

class Rename(object):
    def __init__(self,elements,index=0):
        self.elements=set(elements)
        self.index=index

    def __call__(self,name_i):
        digits=name_i.digits()
        train=(digits[self.index] in self.elements)
        digits[1]=train
        return files.Name("_".join(digits))

def filtr_outlines(in_path,out_path,k=5):
    raw_feats=dtw.read(in_path) 

    train,test=raw_feats.split()
    inliners=get_knn(train,k)
    raw_feats= raw_feats.subset(inliners)
    raise Exception(len(raw_feats))
    new_feats=get_maping(train,errors,names)
    new_feats.append(test)
    new_feats.save(out_path)

def get_knn(train,k=5):
    inliners=[]
    for name_i in train.keys():
        print(name_i)
        near_i=[ name_j.get_cat()==name_i.get_cat()
            for name_j in train.neighbors(name_i,k)]
        if(all(near_i)):
            inliners.append(name_i)
    return inliners

in_path="dtw/test_0"
filtr_outlines(in_path,"smooth4")