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
    for i in range(len(train)):
        all_errors=[error(train_i,weights)
                        for train_i in train]
        active_cls=1-alphas
        active_cls[active_cls<1.0]=0.0
        all_errors= all_errors*active_cls
        t=np.argmax(all_errors)
        err_t=all_errors[t]
        if(err_t<0.01):
            err_t=0.01
        alphas[t]=np.log( (1.0-err_t)/err_t)+np.log(K-1)
        weights=recompute_weights(train[t],alphas[t],weights)
    return alphas

def error(train_i,weights):
	X,y=train_i.X,train_i.get_labels()
	return sum([weights[j]*(1.0-x_i[y[j]]) for j,x_i in enumerate(X)])

def init_weigts(train):
    weights=np.ones((len(train),))
    weights/=weights.shape[0]
    return weights

def recompute_weights(data_t,alpha_t,weights):
    X,y=data_t.X,data_t.get_labels()
    weights=[ weight_j* np.exp( alpha_t* data_t.X[j,y[j]]) 
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