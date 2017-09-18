from sets import Set
import exper,dataset,eval

def ensemble_dataset(basic_paths,adapt_paths,rest_sets):
    basic_data=basic_dataset(basic_paths)
    ensemble_datasets=adapt_datasets(adapt_paths,rest_sets)
    return [ basic_data+data_i
                for data_i in ensemble_datasets]

def basic_dataset(paths,select=150):
    data=exper.single_dataset(paths)
    data=exper.lasso_selection(data,select,norm=True)
    return data

def adapt_datasets(paths,rest_sets,n_feats=30):
    data=[adapt_data(n_feats,path_i,rest_set_i)
            for path_i,rest_set_i in zip(paths,rest_sets)]
    return data        

def adapt_data(n_feats,path_i,restr_set):
    if(type(path_i)!=list):
        path_i=[path_i]
    data=exper.single_dataset(path_i)
    if(restr_set!=None):
        data=restrict_cats(data,restr_set)
    if(n_feats!=None):
        data=exper.lasso_selection(data,n_feats,norm=True)
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
    basic_paths="conf/no_deep.txt"
    adapt_paths=["../AArtyk/binary/cat9/dtw_feats.txt",
                 "../AArtyk/binary/cat11/dtw_feats.txt"]
    rest_sets=[[9],[11]]             
    ensemble_dataset(basic_paths,adapt_paths,rest_sets)