import numpy as np
import feats,exper.cats

def ada_boost(votes_path,data="test"):
    votes=feats.read_list(votes_path)
    y_pred,y_true,names=exper.cats.simple_voting(votes)
    train,test=zip(*[vote_i.split() for vote_i in votes])
    train=[ exper.cats.binarize(train_i) for train_i in train]
    samme(train)

def samme(train):
    weights=init_weigts(train[0])
    errors_t=[error(train_i,weights)
                for train_i in train]
    print(errors_t)

def error(train_i,weights):
	X,y=train_i.X,train_i.get_labels()
	return sum([weights[j]*x_i[y[j]] for j,x_i in enumerate(X)])

def init_weigts(train):
    weights=np.ones((len(train),))
    weights/=weights.shape[0]
    return weights

