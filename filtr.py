import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from collections import defaultdict
import feats,dtw,files

class AllResults(object):
    def __init__(self):
        self.keys=[]
        self.results=[]
    
    def filtr(self,value=None,k=0):
        if(value is None):
            return range(len(self.keys))
        return [i for i,key_i in enumerate(self.keys)
                    if(key_i[k]==value)]
    
    def add(self,key_i,result_i):
        self.keys.append(key_i)
        self.results.append(result_i)

    def to_lines(self,indexes,fun):
        lines=[]
        for i in indexes:
            out_i=get_line(fun(self.results[i]))
            key_i=get_line(self.keys[i])
            lines.append(f"{key_i},{out_i}")
        return lines

def get_line(raw):
    return ",".join([ round(raw_i) for raw_i in raw])

def round(raw,digits=4):
    if(type(raw)==int):
        return f'{raw}'
    return  f"{raw:0.4f}" if type(raw)!=str else raw

def to_csv(lines):
    return "\n".join(lines)

def make_all_results(raw):
    all_results=AllResults()
    for path_i,output in raw:
        for  key_j,result_j in zip(*output):
            all_results.add(key_j,result_j)
    return all_results

class Rename(object):
    def __init__(self,elements,index=2):
        self.elements=set(elements)
        self.index=index

    def __call__(self,name_i,i=None):
        digits=name_i.split('_')#digits()
        train= (digits[self.index] in self.elements)
        digits[1]=str(int(train))
        return files.Name("_".join(digits))

class Subsample(object):
    def __init__(self,k,n):
        self.k=k
        self.n=n
        self.i=0

    def __call__(self,name_i):
        digits=name_i.split('_')
        train= ((self.i % self.n)==self.k)
        digits[1]=str(int(train))
        self.i+=1
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

@files.dir_function(args=1,with_path=True)
def exp(in_path):
    raw_feats=dtw.read(in_path) 
    splits=[
            Subsample(3,8),
            Subsample(4,8),
            Subsample(5,8),
            Subsample(6,8),
            Rename(['1','2'],3),
            Rename(['1','2','5'],3),
            Rename(['1','2','3','5'],3)]
    keys,results=[],[]#AllResults()
    for i,split_i in enumerate(splits):
        data_i=raw_feats.rename(split_i)
        result_i=dtw.test_dtw(data_i)
        keys.append([in_path,i])
        results.append(result_i)
    return (keys,results)


if __name__ == "__main__":
    in_path="dtw"
    results=make_all_results(exp(in_path))
    indexes = results.filtr(value=None,k=1)
    lines=results.to_lines(indexes,fun=lambda x:x.metrics()[:-1])
    print(to_csv(lines))