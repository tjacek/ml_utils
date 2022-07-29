import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from collections import defaultdict
import feats

class Rename(object):
    def __init__(self,elements,index=0):
        self.elements=set(elements)
        self.index=index

    def __call__(self,name_i):
        digits=name_i.digits()
        train=(digits[self.index] in self.elements)
        digits[1]=train
        return  files.Name("_".join(digits))

#def filtr_outlines(in_path,out_path,k=3):
#    raw_feats=feats.read(in_path)[0]
#    train,test=raw_feats.split()
#    errors=get_knn(train,test,k)
#    new_feats=get_maping(train,errors,names)
#    new_feats.append(test)
#    new_feats.save(out_path)

#def filtr_wrong(in_path,out_path,k=3):
#    raw_feats=feats.read(in_path)[0]
#    train,test=raw_feats.split()
#    errors=get_knn(test,train,k)
#    new_feats=feats.Feats()
#    for name_i,data_i in raw_feats.items():
#        if(not name_i in errors):
#            new_feats[name_i]=data_i
#    new_feats.save(out_path)
#    train,test= new_feats.split()
#    print(len(train))

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

def get_knn(train,test,k=3):
    neigh = KNeighborsClassifier(n_neighbors=k)
    X_train,y_train,names =train.as_dataset()
    neigh.fit(X_train, y_train)
    X_test,y_test,names =test.as_dataset()
    y_pred=neigh.predict(X_test)
    errors=set(find_errors(y_test,y_pred,names))
    return errors

def find_errors(y_true,y_pred,names):
    errors=[]	
    for i,true_i in enumerate(y_true):
        if(true_i!=y_pred[i]):
            errors.append(names[i])
    return errors


in_path="../3DHOI/1D_CNN/feats"
filtr_wrong(in_path,"smooth4")