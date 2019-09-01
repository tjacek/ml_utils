from sklearn.metrics import accuracy_score
from sets import Set
import filtr,feats,exper.voting

def exper_voting(in_path,restr,args):
    datasets=exper.voting.get_datasets(**args)

def exper_single(in_path,restr,clf_type="SVC",n_select=None):
    feat_dataset=feats.read(in_path)
    feat_dataset.norm()
    subset=[by_cats(feat_dataset,restr_i) for restr_i in restr] 
    result=[predict_labels(subset_i,clf_type,n_select)  for subset_i in subset]
    acc=[accuracy_score(result[0],result[1]) for result_i in result]
    return acc
#    y_pred,y_true,names=predict_labels(feat_dataset,clf_type,n_select)
#    print(classification_report(y_true, y_pred,digits=4))

def rest_pred(feat_dataset,restr_j,clf_type="LR"):
    subset=[by_cats(feat_dataset,restr_j) for restr_j in restr] 
    return [predict_labels(subset_i,clf_type) for subset_i in subset]

def by_cats(feat_set,cat_set):
    cat_set=Set(cat_set)
    def cat_selector(name_i):
        cat_i=int(name_i.split('_')[0])
        return (cat_i in cat_set)
    train,test=filtr.split(names,cat_selector)
    cat_dict=filtr.filtered_dict(train,feat_set.to_dict())
    return feats.from_dict(cat_dict)