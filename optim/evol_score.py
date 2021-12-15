import sys
sys.path.append("..")
import numpy as np
from scipy.optimize import differential_evolution
import selection,pref,ens

class GenAlg(object):
    def __init__(self,maxiter=10):
        self.maxiter=maxiter

    def __call__(self,loss_fun,n_cand,maxiter=50):
        bound_w = [(0.0, n_cand)  for _ in range(n_cand)]
        result = differential_evolution(loss_fun, bound_w, 
#        init=init_score(15,n_cand),
                 maxiter=self.maxiter, tol=1e-7)
        return result['x']

def exp(paths,n_iters):
    optim=GenAlg()
    old_results,new_results=[],[]
    for i in range(n_iters):
        old_i,s_clf=selection.select_clfs(paths,read_type=None)
        new_i,score_i=find_score(paths,s_clf,optim)
        old_results.append(old_i)
        new_results.append(new_i)
    acc=[result_i.get_acc() for result_i in new_results]
    k=np.argmax(acc)
    new_results[k].report()
    print(old_results[k].get_acc())

def find_score(paths,ensemble,optim):
    ensemble=ens.get_ensemble_helper(ensemble)
    result,votes=ensemble(paths)
    pref_dict=pref.to_pref(votes.results)
    n_cand=pref_dict.n_cand()
    def loss_fun(score):
        result=eval_score(score,n_cand,pref_dict)
        acc=result.get_acc()
        print(acc)
        return (1.0-acc)
    score=optim(loss_fun,n_cand)#,maxiter=10)
    print(score)
    result=eval_score(score,n_cand,pref_dict)
    return result,score

def eval_score(score,n_cand,pref_dict):
    def system_i(name_i,pref_dict):
        return pref.score_rule(name_i,pref_dict, n_cand,score)
    names=pref_dict.keys()
    return pref.election(names,system_i,pref_dict)

if __name__ == "__main__":
    path="../../VCIP/3DHOI/%s/feats"
    common=[path % "1D_CNN",#"../../deep_dtw/dtw"]
                path % "shapelets"]
    binary=path % "ens/splitI/"
    paths=(common,binary)
    exp(paths,10)