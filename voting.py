import numpy as np
from sklearn.metrics import classification_report
import files,feats,exper

def voting(hc_path,deep_paths):
    datasets=get_datasets(hc_path,deep_paths)
    votes=[ exper.predict_labels(data_i,clf_type="SVC") for data_i in datasets]
    y_true=votes[0][1]
    y_ens=np.array([vote_i[0] for vote_i in votes]).T    
    y_pred=[np.argmax(np.bincount(vote_i)) for vote_i in y_ens]    
    print(classification_report(y_true, y_pred,digits=4))

def get_datasets(hc_path,deep_paths):
    hc_feats,full_feats=feats.read(hc_path),[]
    for path_i in files.top_files(deep_paths):
        deep_feats_i=feats.read(path_i)
        full_feats.append( hc_feats +deep_feats_i)
    return full_feats

voting('datasets/exp','deep')