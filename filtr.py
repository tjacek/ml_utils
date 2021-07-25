from sklearn.neighbors import KNeighborsClassifier
from collections import defaultdict
import feats

def filtr_outlines(in_path,k=3):
    raw_feats=feats.read(in_path)[0]
    train,test=raw_feats.split()
    neigh = KNeighborsClassifier(n_neighbors=k)
    X,y,names =train.as_dataset()
    neigh.fit(X, y)
    y_pred=neigh.predict(X)
    errors=find_errors(y,y_pred,names)
    cat_dict=by_cat(names)
    print(cat_dict)

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
filtr_outlines(in_path)