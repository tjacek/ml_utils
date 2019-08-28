import numpy as np
import feats,filtr,exper.persons,exper.voting

def make_cat_feats(args,out_path,clf_type="LR"):
    datasets=exper.voting.get_datasets(**args)
    pred_dicts=[pred_dict(data_i,clf_type) 
                    for data_i in datasets]
    names=pred_dicts[0].keys()
    names.sort()
    def cat_helper(name_i):
        return np.array([pred_j[name_i] for pred_j in pred_dicts])
    pred_feats={name_i:cat_helper(name_i) for name_i in names}
    pred_feats=feats.from_dict(pred_feats)
    pred_feats.save(out_path)

def pred_dict(data_i,clf_type="LR"):
    train,test=filtr.split(data_i.info)
    train_data=filtr.filtered_dict(train,data_i)
    by_person=exper.persons.samples_by_person(train)
    person_pred=exper.persons.pred_by_person(train_data,by_person,clf_type)
    pairs=[]
    for person_i,pred_i in person_pred.items():
        one,y_one,y_pred=pred_i
        pairs+=zip(one,y_pred)
    return dict(pairs)