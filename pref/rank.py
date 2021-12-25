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
    def helper(cat_i,clf_j):
        names=list(votes_dict.keys())
        values=[ votes_k[clf_j][cat_i] 
                   for name_k,votes_k in votes_dict.items()]
        return [(names[t],t) for t in np.flip(np.argsort(values))]
    print(helper(0,0))
#def get_ranks(paths):
#    common=paths[0]
#    dataset= ens.read_dataset(common,None)[0]
#    result=learn.train_model(dataset,clf_type="LR")
#    rank_dict=to_rank(result)
#    pairs_transform(rank_dict,dataset)

#def to_rank(result):
#	return { name_i:np.flip(np.argsort(y_i)) 
#		for name_i,y_i in zip(result.names,result.y_pred)}

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
    binary=path % "ens/splitII/"
    paths=(common,binary)
    get_ranks(paths)