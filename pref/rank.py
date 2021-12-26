import sys
sys.path.append("..")
import itertools
import numpy as np
#from scipy import stats
#import pylab as pl
#from sklearn import svm, linear_model
import ens#,learn

def get_ranks(paths):
    ensemble=ens.get_ensemble_helper(ensemble=None)
    result,votes=ensemble(paths)
    votes_dict= votes.as_dict()
    n_cats=result.n_cats()
#    check(votes_dict,n_cats)
    def helper(cat_i,clf_j):
        names=list(votes_dict.keys())
        values=[ votes_dict[name_k][clf_j][cat_i] 
                   for name_k in names]
        return dict([(names[t],t) 
            for t in np.flip(np.argsort(values))])
    best_dicts=[helper(0,clf_i) for clf_i in range(n_cats)]
    pref_dict={ name_i:[dict_j[name_i] 
                        for dict_j in best_dicts]  
                            for name_i in best_dicts[0]}
    print(pref_dict)

def check(votes_dict,n_cats):
    test_name=list(votes_dict.keys())[0]
    comp_rank=[votes_dict[test_name][clf_j][0]
                for clf_j in range(n_cats)]
    raise Exception(comp_rank)


#def pairs_transform(rank_dict,dataset):
#    X,y=[],[]
#    for name_i in rank_dict.keys():
#    	X.append(dataset[name_i])
#    	y.append(rank_dict[name_i])
#    comb = itertools.combinations(range(len(y)),2)
#    Xp,yp=[],[]
#    for a_i,b_i in comb:
#    	print(len(X))
#    	raise Exception(X[a_i].shape)
#    print(list(comb))

if __name__ == "__main__":
    path="../../VCIP/3DHOI/%s/feats"
    common=[path % "1D_CNN",#"../deep_dtw/dtw"]
         path % "shapelets"]
    common=None
    binary=path % "ens/splitII/"
    paths=(common,binary)
    get_ranks(paths)