import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from collections import defaultdict
import feats,dtw,files

class Rename(object):
    def __init__(self,elements,index=2):
        self.elements=set(elements)
        self.index=index

    def __call__(self,name_i):
        digits=name_i.split('_')#digits()
        train=(digits[self.index] in self.elements)
        digits[1]=str(int(train))
        return files.Name("_".join(digits))

def filtr_outlines(in_path,out_path,k=5):
    raw_feats=dtw.read(in_path) 
    train,test=raw_feats.split()
    inliners=get_knn(train,k)
    raw_feats= raw_feats.subset(inliners)
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

def exp(in_path):
    raw_feats=dtw.read(in_path) 
    print(raw_feats.keys())
    splits=[Rename(['1','2'],3),
            Rename(['1','2','5'],3),
            Rename(['1','2','3','5'],3)]
    for split_i in splits:
        raw_feats.check()
        data_i=raw_feats.rename(split_i)
        data_i.check()
        result_i=dtw.test_dtw(data_i)
        print(result_i.get_acc())

#        train,test=data_i.split()
#        print(len(train))
#        print(len(test))

if __name__ == "__main__":
    in_path="dtw/test_0"
    exp(in_path)#,"smooth4")