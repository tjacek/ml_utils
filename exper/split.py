import numpy as np
import exper,exper.persons
import feats,filtr,learn

def split_ensemble(args,clf_type="LR",restr=None,show=False):
    datasets=exper.voting.get_datasets(**args)
    datasets=restr_dataset(datasets,restr)
    result=[train_cls(list(data_i),clf_type) for data_i in datasets]
    if(show):
        for result_i in result:
            learn.show_result(*result_i) 
    scores=[ learn.compute_score(result_i[1],result_i[0],as_str=False)
                for result_i in result]
    scores=np.array(scores)
    print(np.mean(scores,axis=0))

def train_cls(datasets,clf_type="SVC"):
    results=[]
    for data_i in datasets:
        train,test=data_i.split()
        pairs=exper.persons.pred_vectors(train,test,clf_type)
        results.append(dict(pairs))
    names=list(results[0].keys())
    votes={name_i: [result_i[name_i] for result_i in results]
                for name_i in names}
    y_true=filtr.all_cats(names)
    y_pred=[ voting(vote_i)
            for name_i,vote_i in votes.items()]
    return y_pred,y_true,names

def voting(votes):
    dist=np.sum(votes,axis=0)
    return np.argmax(dist)

def restr_dataset(datasets,restr):
    subset=[[by_cats(data_i,restr_j) 
                for restr_j in restr]
                    for data_i in datasets] 
    return list(zip(*subset))

def by_cats(feat_set,cat_set):
    cat_set=set(cat_set)
    def cat_selector(name_i):
        cat_i=int(name_i.split('_')[0])
        return (cat_i in cat_set)
    train,test=filtr.split(feat_set.info,cat_selector)
    cat_dict=filtr.filtered_dict(train,feat_set.to_dict())
    cat_dict=filtr.ordered_cats(cat_dict)
    return feats.from_dict(cat_dict)