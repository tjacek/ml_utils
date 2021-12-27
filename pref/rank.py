import sys
sys.path.append("..")
import itertools
import numpy as np
#from scipy import stats
#import pylab as pl
#from sklearn import svm, linear_model
import ens#,learn

class VotesDict(object):
    def __init__(self,votes_dict):
        self.votes_dict=votes_dict
        self.names=list(self.votes_dict.keys())

    def get_values(self,clf_i,cat_j):
        values=[ self.votes_dict[name_k][clf_i][cat_j] 
                   for name_k in self.names]
        return np.array(values)

    def get_order(self,clf_i,cat_j):
        values=self.get_values(clf_i,cat_j)
        return np.flip(np.argsort(values))

    def get_dict(self,clf_i,cat_j):
        return {self.names[y]:x 
            for x,y in enumerate(self.get_order(clf_i,cat_j))}

def get_ranks(paths):
    ensemble=ens.get_ensemble_helper(ensemble=None)
    result,votes=ensemble(paths)
    votes_dict= VotesDict(votes.as_dict())
    n_cats=result.n_cats()
    best_dicts=[votes_dict.get_dict(clf_i,0) for clf_i in range(n_cats)]
    pref_dict={ name_i:[dict_j[name_i] 
                        for dict_j in best_dicts]  
                            for name_i in votes_dict.names}
    print(pref_dict)

if __name__ == "__main__":
    path="../../VCIP/3DHOI/%s/feats"
    common=[path % "1D_CNN",#"../deep_dtw/dtw"]
         path % "shapelets"]
    common=None
    binary=path % "ens/splitII/"
    paths=(common,binary)
    get_ranks(paths)