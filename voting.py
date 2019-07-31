import numpy as np
from sklearn.metrics import classification_report
import files,feats,exper,learn

class Ensemble(object):
    def __init__(self,clf_type="LR"):
        self.clf_type=clf_type

    def __call__(self,hc_path,deep_paths,n_feats=None):
        datasets=get_datasets(hc_path,deep_paths,n_feats)
        votes=[ self.predict(data_i) for data_i in datasets]
        y_true=votes[0][1]
        y_ens=np.array([vote_i[0] for vote_i in votes]).T    
        y_pred=[np.argmax(np.bincount(vote_i)) for vote_i in y_ens]    
        print(classification_report(y_true, y_pred,digits=4))
        return learn.compute_score(y_true,y_pred,as_str=True)

    def predict(self,data_i):
        return exper.predict_labels(data_i,clf_type=self.clf_type) 

def get_datasets(hc_path,deep_paths,n_feats):
    hc_feats=read_hc(hc_path,n_feats)
    if(not deep_paths):
        return [hc_feats]    
    full_feats=[]
    for path_i in files.top_files(deep_paths):
        deep_i=feats.read(path_i)
        deep_i.norm()
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

voting=Ensemble("SVC")
print(voting('../AA/hand','../AA/s_deep',100))