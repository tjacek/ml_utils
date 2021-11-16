import numpy as np
import learn,feats,script

class Ensemble(object):
    def __init__(self,read=None):#,transform=None):
        if(read is None):
            read=read_dataset
#        self.transform=transform
        self.read=read

    def __call__(self,paths,binary=False,clf="LR",s_clf=None):
        datasets=self.get_datasets(paths)
        votes=make_votes(datasets,clf=clf)
        if(s_clf):
            votes=Votes([votes.results[i] for i in s_clf])
        result=votes.voting(binary)
        print(result.get_acc()) 
        return result,votes

    def get_datasets(self,paths):
        if(type(paths)==dict):
            common,binary=paths["common"],paths["binary"]
        else:
            common,binary=paths    
        datasets=self.read(common,binary)
#        if(self.transform):
#            datasets=[self.transform(data_i)  for data_i in datasets]
        return datasets

class Votes(object):
    def __init__(self,results):
        self.results=results

    def __len__(self):
        return len(self.results)

    def voting(self,binary=False):
        if(binary):
            votes=np.array([ result_i.as_hard_votes() 
                    for result_i in self.results])
        else:
            votes=np.array([ result_i.as_numpy() 
                    for result_i in self.results])
        votes=np.sum(votes,axis=0)
        return learn.Result(self.results[0].y_true,votes,self.results[0].names)

    def weighted(self,weights):
        votes=np.array([ weight_i*result_i.as_numpy() 
                    for weight_i,result_i in zip(weights,self.results)])
        votes=np.sum(votes,axis=0)
        return learn.Result(self.results[0].y_true,votes,self.results[0].names)

    def get_acc(self):
        return [ result_i.get_acc() for result_i in self.results]

class EnsembleHelper(object):
    def __init__(self,ensemble,binary=False,clf="LR",s_clf=None):
        self.ensemble=ensemble
        self.binary=binary
        self.clf=clf
        self.s_clf=s_clf

    def __call__(self,paths):
        return self.ensemble(paths,binary=self.binary,
                    clf=self.clf,s_clf=self.s_clf)

def read_dataset(common_path,deep_path):
    if(not common_path):
        return read_deep(deep_path)
    if(not deep_path):
        return feats.read(common_path)
    common_data=feats.read(common_path)[0]
    deep_data=read_deep(deep_path)
#    datasets=[common_data+ data_i 
#                for data_i in deep_data]
    datasets=[ feats.agum_concat(common_data,data_i) 
                for data_i in deep_data]
    return datasets

def read_deep(deep_path):
    if(type(deep_path)==list):
        datasets=[]
        for deep_i in deep_path:
            datasets+=feats.read(deep_i)
        return datasets
    return feats.read(deep_path)

def get_models(datasets,clf="LR",model_only=True):
    if(type(datasets)==dict):
        datasets=read_dataset(datasets["common"],datasets["binary"])
    models=[learn.train_model(data_i,clf_type=clf,model_only=model_only) 
                for data_i in datasets]
    return models,datasets
    
def make_votes(datasets,clf="LR"):    
    results=[learn.train_model(data_i,clf_type=clf,binary=False)
                    for data_i in datasets]
    return Votes(results)  

def read_multi(common_path,deep_path):
    if( type(common_path)!=list):
        common_path=[common_path]
    deep_data=read_deep(deep_path)
    datasets=[]
    for path_i in common_path:
        common_i=feats.read(path_i)[0]
        for deep_j in deep_data:
            datasets.append(common_i+deep_j)
    return datasets

if __name__ == "__main__":
    ensemble=Ensemble()#read_multi)
    common_path="../common/feats.txt"
    paths={'common':None,'binary':"../ens/_feats"}
    result,votes=ensemble(paths,clf="LR",binary=False)
    result.report()
    print(result.get_cf())
    errors=result.get_errors()
    print(learn.order_error(errors))