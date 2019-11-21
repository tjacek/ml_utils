import numpy as np
import feats

def sparse_feats(in_path):
    data=feats.read(in_path)
    weight=np.sum(data.X,axis=0)
    weight/=np.std(weight)
    print(weight)
