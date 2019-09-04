import numpy as np
import feats,filtr,exper.persons,exper.voting

def make_cat_feats(args,out_path,clf_type="LR",binary=True):
    datasets=exper.voting.get_datasets(**args)
    pred_dicts=[pred_dict(data_i,clf_type) 
                    for data_i in datasets]
    names=pred_dicts[0].keys()
    names.sort()
    def cat_helper(name_i):
        return np.array([pred_j[name_i] for pred_j in pred_dicts])
    pred_feats={name_i:cat_helper(name_i) for name_i in names}
    pred_feats=feats.from_dict(pred_feats)
    if(binary):
    	pred_feats=binary_dataset(pred_feats)
    pred_feats.save(out_path)

def in_sample(data_i,clf_type="LR"):
    train,test=filtr.split(data_i.info)
    train_data=filtr.filtered_dict(train,data_i)
    by_person=exper.persons.samples_by_person(train)
    person_pred=exper.persons.pred_by_person(train_data,by_person,clf_type)
    pairs=[]
    for person_i,pred_i in person_pred.items():
        one,y_one,y_pred=pred_i
        pairs+=zip(one,y_pred)
    return pairs

def pred_dict(data_i,clf_type="LR"):
    y_pred,y_true,names=exper.predict_labels(data_i,clf_type)
    test_pred_i=zip(names,[y_i-1 for y_i in y_pred])
    train_pred=in_sample(data_i,clf_type)
    return dict(test_pred_i+train_pred)

def binary_dataset(data_i):
    n_cats=int(np.amax(data_i.X))
    def helper(x_i):
    	binary_vec=[]
        for cat_j in x_i:
            one_hot=np.zeros((n_cats+1,))
            one_hot[int(cat_j)]=1
            binary_vec+=list(one_hot)
        return np.array(binary_vec)
    new_X=np.array([helper(x_i) 
    	        for x_i in data_i.X])
    return feats.FeatureSet(new_X,data_i.info)

def from_binary(in_path):
    binary_data=feats.read(in_path)
    n_cats= filtr.n_cats(binary_data.info)
    n_clfs=binary_data.dim()/n_cats
    def binary_helper(x_i):
        one_hot=[ x_i[j*n_cats:(j+1)*n_cats] for j in range(n_clfs)]
        return [ np.argmax(vec_j) for vec_j in one_hot]
    binary_data=binary_data.to_dict()
    return { name_i:binary_helper(x_i) 
                for name_i,x_i in binary_data.items()}