import numpy as np
from sklearn.metrics import accuracy_score
import feats, exper.cats#,check

def random_ensemble(in_path,n_cats=20,n=100):
    votes=feats.read_list(in_path)
    for i in range(2,n_cats):
        result=[subsample(votes,k=i) for j in range(n)]
        stats=np.amax(result),np.mean(result),np.median(result),np.amin(result)
        print("%s,%s,%s,%s" % stats)

def subsample(votes,k=5):
    samples=np.random.choice(votes,size=k,replace=True)
    result=exper.cats.simple_voting(samples)
    acc=accuracy_score(result[0],result[1])
    return acc

path_i="proj2/ens5/LR/stats_basic"

random_ensemble(path_i,20)