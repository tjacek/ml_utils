from sets import Set
import exper,dataset,eval,voting

def ensemble_dataset(basic_paths,adapt_paths):
    basic_data=basic_dataset(basic_paths)
    ensemble_datasets=adapt_datasets(adapt_paths,rest_sets)
    return [data_i + basic_data
                for data_i in ensemble_datasets]

def basic_dataset(paths,n_feats=150):
    data=exper.single_dataset(paths)
    if(type(n_feats)==int):
        n_feats=('rfe',n_feats)
    data=exper.feat_selection(data,n_feats,norm=True)
    return data

def adapt_datasets(paths,rest_sets,n_feats=50):
    if(len(paths)!=len(rest_sets)):
        raise Exception("paths and rest_sets have different lenght")    
    if(type(n_feats)==int):
        n_feats=('rfe',n_feats)
    data=[adapt_data( n_feats,path_i,rest_set_i)
            for path_i,rest_set_i in zip(paths,rest_sets)]
    return data        

def adapt_data(n_feats,path_i):
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
    basic_paths="conf/no_deep2.txt"  
    nn_path="../AArtyk3/feats/nn_"
    adapt_paths=[nn_path+str(i+1)
                    for i in range(19)]

    #rest_sets=[16,25,26]
    datasets=ensemble_dataset(basic_paths,adapt_paths)
    ensemble=voting.get_ensemble('lr')
    ensemble_pred,y_true=ensemble(datasets)
    voting.show_result(ensemble_pred,y_true,conf=True)