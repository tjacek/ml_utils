import numpy as np
import exper,exper.persons
import feats,filtr

def split_ensemble(args,clf_type="LR",restr=None):
    datasets=exper.voting.get_datasets(**args)
    datasets=restr_dataset(datasets,restr)
    print(len(datasets))
    acc=[train_cls(list(data_i),clf_type) for data_i in datasets]
    print(acc)
    print(np.mean(acc))

def train_cls(datasets,clf_type="SVC"):
    results=[]
    for data_i in datasets:
        train,test=data_i.split()
        pairs=exper.persons.pred_vectors(train,test,clf_type)
        results.append(dict(pairs))
    names=results[0].keys()
    votes={name_i: [result_i[name_i] for result_i in results]
                for name_i in names}
    pred=[ correc_pred(name_i,vote_i)
            for name_i,vote_i in votes.items()]
    return np.mean(pred)

def correc_pred(name_i,votes):
    cat_i=int(name_i.split('_')[0])-1
    dist=np.sum(votes,axis=0)
    pred_i=np.argmax(dist)
    return float(cat_i==pred_i)

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