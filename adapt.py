from sets import Set
import exper,dataset,eval,voting,learn

class EnsembleDataset(object):
    def __init__(self, basic_feats,adapt_feats):
        if(type(basic_feats)==int):
            basic_feats=('lasso',n_feats)
        if(type(adapt_feats)==int):
            adapt_feats=('lasso',n_feats)
        self.basic_feats=basic_feats
        self.adapt_feats=adapt_feats

    def __call__(self,basic_paths,adapt_paths):
        basic_data=self.basic_dataset(basic_paths)
        if(len(adapt_paths)==0):
            return [basic_data]
        ensemble_datasets=self.adapt_datasets(adapt_paths)
        return [data_i + basic_data
                for data_i in ensemble_datasets]

    def basic_dataset(self,paths):
        data=exper.single_dataset(paths)
        data=exper.feat_selection(data,self.basic_feats,norm=True)
        return data

    def adapt_datasets(self,paths):  
        data=[adapt_data(self.adapt_feats,path_i)
                for path_i in zip(paths)]
        return data        

def adapt_data(n_feats,path_i):
    print(path_i)
    if(type(path_i)==tuple):
        path_i=list(path_i)
    if(type(path_i)!=list):
        path_i=[path_i]
    data=exper.single_dataset(path_i)
    old_y=data.y
    if(n_feats!=None):
        data=exper.feat_selection(data,n_feats,norm=True)
    data.y=old_y
    return data           

def gen_pahts(nn_path,indices):
    if(type(indices)==int):
        indices=range(indices)
    if(type(indices)==tuple):
        indices=range(indices[0], indices[1])
    return [nn_path+str(i+1)
                    for i in indices]

if __name__ == "__main__":
    basic_paths="conf/no_deep.txt"  
    nn_path="../AArtyk/all_feats/nn_"
    adapt_paths=[nn_path+str(i)
                    for i in range(20)]
    ensemble_dataset=EnsembleDataset(('lasso',150),('lasso',50))
    datasets=ensemble_dataset(basic_paths,adapt_paths)
    ensemble=voting.get_ensemble('lr')
    ensemble_pred,y_true=ensemble(datasets)
    learn.show_result(ensemble_pred,y_true,conf=True)