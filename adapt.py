from sets import Set
import exper,dataset,eval

def get_data(n_feats,paths,restric):
    def helper(i):
        return adapt_data(n_feats[i],paths[i],restric[i])
    all_datasets=[ helper(i) 
                    for i,path_i in enumerate(paths)]
    return dataset.unify_feat(all_datasets)

def adapt_data(n_feats,path_i,restr_set):
    if(type(path_i)!=list):
        path_i=[path_i]
    data=exper.single_dataset(path_i)
    if(restr_set!=None):
        data=restrict_cats(data,restr_set)
    if(n_feats!=None):
        data=exper.lasso_selection(data,n_feats,norm=True)
    print(data==None)
    return data

def restrict_cats(data,restr_set):
    if(type(restr_set)==Set):
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
    n_feats=[None,150]
    paths=['../AArtyk/simple/all/simple.txt',
	       '../AArtyk/simple/max_z/dtw_dataset.txt']
    restric=[None,[i for i in range(20)]]
    data=get_data(n_feats,paths,restric)
    exper.experiment_basic(data,cls_type='svm')