import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from collections import defaultdict
import feats

def filtr_outlines(in_path,out_path,k=3):
    raw_feats=feats.read(in_path)[0]
    train,test=raw_feats.split()
    neigh = KNeighborsClassifier(n_neighbors=k)
    X,y,names =train.as_dataset()
    neigh.fit(X, y)
    y_pred=neigh.predict(X)
    errors=set(find_errors(y,y_pred,names))
    new_feats=get_maping(train,errors,names)
    new_feats.append(test)
    new_feats.save(out_path)

def get_maping(old_feats,errors,names):
    good_names=set(names).difference(errors)
    cat_dict=by_cat(good_names)
    name_map={}
    for name_i in names:
        if( name_i in errors):
            cat_i=cat_dict[name_i.get_cat()]
            distance=[ np.linalg.norm(old_feats[name_i]-old_feats[name_j]) 
                        for name_j in cat_i]
            new_name_i=cat_i[np.argmax(distance)]
            name_map[name_i]=new_name_i
        else:
            name_map[name_i]=name_i
    new_feats={ old_i:old_feats[new_i] 
                for old_i,new_i in name_map.items()}
    return feats.Feats(new_feats)

def find_errors(y_true,y_pred,names):
    errors=[]	
    for i,true_i in enumerate(y_true):
        if(true_i!=y_pred[i]):
            errors.append(names[i])
    return errors

def by_cat(names):
    cat_dict=defaultdict(lambda:[])
    for name_i in names:
    	cat_dict[name_i.get_cat()].append(name_i)
    return cat_dict

in_path="../3DHOI/1D_CNN/feats"
filtr_outlines(in_path,"smooth")