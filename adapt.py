from sets import Set
import exper,dataset,eval,voting

def ensemble_dataset(basic_paths,adapt_paths):
    basic_data=basic_dataset(basic_paths)
    ensemble_datasets=adapt_datasets(adapt_paths)
    return [data_i + basic_data
                for data_i in ensemble_datasets]

def basic_dataset(paths,n_feats=150):
    data=exper.single_dataset(paths)
    if(type(n_feats)==int):
        n_feats=('rfe',n_feats)
    data=exper.feat_selection(data,n_feats,norm=True)
    return data

def adapt_datasets(paths,n_feats=50):  
    if(type(n_feats)==int):
        n_feats=('rfe',n_feats)
    data=[adapt_data( n_feats,path_i)
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
    datasets=ensemble_dataset(basic_paths,adapt_paths)
    ensemble=voting.get_ensemble('lr')
    ensemble_pred,y_true=ensemble(datasets)
    voting.show_result(ensemble_pred,y_true,conf=True)