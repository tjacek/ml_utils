import numpy as np
from sklearn.metrics import accuracy_score
from sets import Set
import filtr,feats,exper.voting

def exper_voting(in_path,restr,args):
    datasets=exper.voting.get_datasets(**args)

def exper_single(args,restr,clf_type="SVC"):
    datasets=exper.voting.get_datasets(**args)
    result=[restr_voting(datasets,restr_i,clf_type) for restr_i in restr]
    acc=[accuracy_score(result_i[0],result_i[1]) for result_i in result]
    mean_acc=np.mean(acc)
    return acc,mean_acc

def restr_voting(datasets,restr,clf_type="LR"):
    subset=[by_cats(data_i,restr) for data_i in datasets] 
    preds=[exper.predict_labels(subset_i,clf_type) for subset_i in subset]
    return exper.voting.count_votes(preds)
#def rest_pred(feat_dataset,restr_j,clf_type="LR"):
#    subset=[by_cats(feat_dataset,restr_j) for restr_j in restr] 
#    return [predict_labels(subset_i,clf_type) for subset_i in subset]

def by_cats(feat_set,cat_set):
    cat_set=Set(cat_set)
    def cat_selector(name_i):
        cat_i=int(name_i.split('_')[0])
        return (cat_i in cat_set)
    train,test=filtr.split(feat_set.info,cat_selector)
    cat_dict=filtr.filtered_dict(train,feat_set.to_dict())
    return feats.from_dict(cat_dict)