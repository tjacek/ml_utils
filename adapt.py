from sets import Set
import exper,dataset,eval,voting

def ensemble_dataset(basic_paths,adapt_paths,rest_sets=None):
    basic_data=basic_dataset(basic_paths)
    if(rest_sets==None):
        rest_sets=[None for path_i in adapt_paths]
    ensemble_datasets=adapt_datasets(adapt_paths,rest_sets)
    return [data_i + basic_data
                for data_i in ensemble_datasets]

def basic_dataset(paths,n_feats=150):
    data=exper.single_dataset(paths)
    data=exper.feat_selection(data,('rfe',n_feats),norm=True)
    return data

def adapt_datasets(paths,rest_sets,n_feats=50):
    if(len(paths)!=len(rest_sets)):
        raise Exception("paths and rest_sets have different lenght")
    data=[adapt_data( ('rfe',n_feats),path_i,rest_set_i)
            for path_i,rest_set_i in zip(paths,rest_sets)]
    return data        

def adapt_data(n_feats,path_i,restr_set):
    if(type(path_i)!=list):
        path_i=[path_i]
    data=exper.single_dataset(path_i)
    old_y=data.y
    if(restr_set!=None):
        data=restrict_cats(data,restr_set)
    if(n_feats!=None):
        data=exper.feat_selection(data,n_feats,norm=True)
    data.y=old_y
    return data

def restrict_cats(data,restr_set):
    if(type(restr_set)==int):
        restr_set=[restr_set]
    if(type(restr_set)!=Set):
        restr_set=Set(restr_set)	
    def helper(cat_i):
        if(cat_i in restr_set):
            return cat_i
        else:
            return 0	
    new_cats=[ helper(cat_i)
                for cat_i in data.y]
    data.y=new_cats
    return data               

if __name__ == "__main__":
    basic_paths="conf/no_deep2.txt"  
    #adapt_paths=["../AArtyk2/deep/16/simple.txt"]
    #adapt_paths=["../AArtyk/binary_time/cat0/simple.txt",
    #             "../AArtyk/binary_time/cat1/simple.txt",
    #             "../AArtyk/binary_time/cat4/simple.txt",
    #             "../AArtyk/binary_time/cat5/simple.txt",
    #             "../AArtyk/binary_time/cat9/simple.txt",
    #             "../AArtyk/binary_time/cat11/simple.txt",
    #             "../AArtyk/binary_time/cat14/simple.txt"]
    #adapt_paths=[adapt_paths[0],adapt_paths[1],adapt_paths[2],
    #             adapt_paths[4],adapt_paths[6]]
    #adapt_paths=[adapt_paths[1],adapt_paths[4],adapt_paths[6]]             
    #rest_sets=[None for path_i in adapt_paths]
    #basic_paths="conf/dummy.txt"  
    adapt_paths=["../AArtyk2/deep/16/simple.txt",
                 "../AArtyk2/deep/25/simple.txt",
                 "../AArtyk2/deep/26/simple.txt"]

    rest_sets=[16,25,26]
    datasets=ensemble_dataset(basic_paths,adapt_paths,rest_sets)
    ensemble=voting.get_ensemble('lr')
    ensemble_pred,y_true=ensemble(datasets)
    voting.show_result(ensemble_pred,y_true,conf=True)