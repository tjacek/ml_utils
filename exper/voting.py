import numpy as np
from sklearn.metrics import classification_report
import files,feats,exper,learn

class Ensemble(object):
    def __init__(self,clf_type="LR"):
        self.clf_type=clf_type

    def __call__(self,hc_path,deep_paths,n_feats=None):
        datasets=get_datasets(hc_path,deep_paths,n_feats)
        votes=predict(datasets,self.clf_type)
        y_true,y_pred=predict(datasets,self.clf_type)
        print(classification_report(y_true, y_pred,digits=4))
#        return learn.compute_score(y_true,y_pred,as_str=True)

def predict(datasets,clf_type):
    votes=[exper.predict_labels(data_i,clf_type)
                for data_i in datasets]
    return count_votes(votes)

def count_votes(votes):
    y_true=votes[0][1]
    y_ens=np.array([vote_i[0] for vote_i in votes]).T    
    y_pred=[np.argmax(np.bincount(vote_i)) for vote_i in y_ens]  
    return y_true,y_pred

def get_datasets(hc_path,deep_paths,n_feats):
    if(not n_feats):
        n_feats=0
    (n_hc_feats,n_deep_feats)= (n_feats,None) if(type(n_feats)==int) else n_feats
    hc_feats=read_hc(hc_path,n_hc_feats)
    if(not deep_paths):
        return [hc_feats]    
    full_feats=[]
    for path_i in files.top_files(deep_paths):
        print(path_i)
        deep_i=feats.read(path_i)
        deep_i.norm()
        if(n_deep_feats):
            deep_i.reduce(n_deep_feats)
        full_i= (hc_feats + deep_i) if(hc_feats) else deep_i
        full_feats.append( full_i)
    return full_feats

def read_hc(hc_path,n_feats):
    if(not hc_path):
        return None
    hc_feats=feats.read(hc_path)
    hc_feats.norm()
    if(n_feats):
        hc_feats.reduce(n_feats)
    return hc_feats


if __name__ == "__main__":
    paths=["exp",'../res_ensemble/feats/basic']
    exper_single(["exp",'../res_ensemble/binary_cats'],"SVC",100)#"../res_ensemble/feats/res1","SVC")
#    paths=files.top_files('../res_ensemble/binary_feats')
#    paths.sort()
#    for path_i in paths:
#        print(path_i)
#        paths=["exp",path_i]
#        exper_single(paths,"SVC",100)