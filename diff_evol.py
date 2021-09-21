import numpy as np
from scipy.optimize import differential_evolution
import ens,learn

class LossFunction(object):
    def __init__(self, votes):
        self.votes=votes
	
    def __call__(self,weights):
        raise Exception(len(weights))	
        norm=weights/np.sum(weights)
        result=self.votes.weighted(norm)
        return 1.0-result.get_acc()

class OptimizeWeights(object):
    def	__init__(self,loss=None,maxiter=1000):
        if(loss is None):
            loss=LossFunction
        self.loss=loss
        self.maxiter=maxiter

    def __call__(self,common,deep,clf="LR"):
        datasets=ens.read_dataset(common,deep)
        weights=self.find_weights(datasets)
        results=learn.train_ens(datasets,clf="LR")
        votes=ens.Votes(results)
        result=votes.weighted(weights)
        return result

    def find_weights(self,datasets):
        results=validation_votes(datasets)
        loss_fun=self.loss(ens.Votes(results))
        bound_w = [(0.0, 1.0)  for _ in datasets]
#        raise Exception(len(results))
        result = differential_evolution(loss_fun, bound_w, maxiter=self.maxiter, tol=1e-7)
        weights=result['x']
        return weights

def validation_votes(datasets,clf="LR"):
    results=[]
    for data_i in datasets:
        data_i.norm()
        train=data_i.split()[0]
        clf_i=learn.make_model(train,clf)
        y_pred=clf_i.predict_proba(train.get_X())
        result_i =learn.Result(train.get_labels(),y_pred,train.names())
        results.append(result_i)
    return results

if __name__ == "__main__":
    dir_path="../3DHOI/"
    binary_path="%s/ens/I/feats" % dir_path
    base_path="%s/1D_CNN/feats" % dir_path
    dtw_path="../deep_dtw/dtw"
    ae_path="../best2/3_layers/feats"
    common=[base_path,dtw_path]
    diff_voting=OptimizeWeights()
    result=diff_voting(common,binary_path)
    result.report()