import numpy as np
from sklearn.metrics import classification_report
import files,feats,exper

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

    def predict(self,data_i):
        return exper.predict_labels(data_i,clf_type=self.clf_type) 

def get_datasets(hc_path,deep_paths,n_feats):
    hc_feats,full_feats=feats.read(hc_path),[]
    if(n_feats):
        hc_feats.norm()
        hc_feats.reduce(n_feats)
    for path_i in files.top_files(deep_paths):
        deep_feats_i=feats.read(path_i)
        deep_feats_i.norm()
        full_feats.append( hc_feats +deep_feats_i)
    return full_feats

voting=Ensemble("SVC")
voting('datasets/exp','deep',100)