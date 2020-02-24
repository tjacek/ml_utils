import numpy as np
import feats,exper.cats
from sklearn.metrics import classification_report

def ada_boost(votes_path,show=True):
    votes=feats.read_list(votes_path)
    y_pred,y_true,names=exper.cats.simple_voting(votes)
    train,test=zip(*[vote_i.split() for vote_i in votes])
    train=[ exper.cats.binarize(train_i) for train_i in train]
    alphas=samme(train)
    y_pred=weighted_voting(test,alphas)
    if(show):
        print(classification_report(y_true,y_pred,digits=4))
    return y_true,y_pred,names

def samme(train):
    K=train[0].n_cats()
    alphas=np.zeros((K,))
    weights=init_weigts(train[0])
    active_cls=np.zeros((K,))
    for i in range(len(train)):
        all_errors=np.array([error(train_i,weights)
                                    for train_i in train])
        all_errors[active_cls>0.0]=np.inf
        t=np.argmin(all_errors)
        active_cls[t]=1.0
        err_t=all_errors[t]
        if(err_t<0.01):
            err_t=0.01
        alphas[t]=np.log((1.0-err_t)/err_t)+np.log(K-1)
        weights=recompute_weights(train[t],alphas[t],weights)
    print(alphas)
    return alphas

def error(train_i,weights):
	X,y=train_i.X,train_i.get_labels()
	return sum([weights[j]*(1.0-x_i[y[j]]) for j,x_i in enumerate(X)])

def init_weigts(train_i):
    weights=np.ones((len(train_i),))
    weights/=weights.shape[0]
    return weights

def recompute_weights(data_t,alpha_t,weights):
    X,y=data_t.X,data_t.get_labels()
    weights=[ weight_j* np.exp( alpha_t* (1.0- data_t.X[j,y[j]])) 
                for j,weight_j in enumerate(weights)]
    weights=np.array(weights)
    return weights/np.sum(weights)

def weighted_voting(votes,alphas):
    def vote_helper(vote_i):
        vote_i=np.array([alphas[j]*vote_ij 
                    for j,vote_ij in enumerate(vote_i)])	
        vote_i=np.sum(vote_i,axis=0)
        return np.argmax(vote_i)
    X=np.array([vote_i.X for vote_i in votes])
    X=np.swapaxes(X,0,1)
    return [vote_helper(vote_i) for vote_i in X]