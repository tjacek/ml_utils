import sys
sys.path.append("..")
import numpy as np
from scipy.optimize import differential_evolution
import ens,pref

def find_score(paths,ensemble=None):
    ensemble=ens.get_ensemble_helper(ensemble)
    result,votes=ensemble(paths)
    pref_dict=pref.to_pref(votes.results)
    n_cand=pref_dict.n_cand()
    bound_w = [(0.0, n_cand)  for _ in range(n_cand)]
    def loss_fun(score):
        result=eval_score(score,n_cand,pref_dict)
        acc=result.get_acc()
        print(acc)
        return (1.0-acc)
    result = differential_evolution(loss_fun, bound_w, #init=init_score(15,n_cand),
        maxiter=50, tol=1e-7)
    score=result['x']
    result=eval_score(score,n_cand,pref_dict)
    result.report()
    print(score)

def eval_score(score,n_cand,pref_dict):
    def system_i(name_i,pref_dict):
        return pref.score_rule(name_i,pref_dict, n_cand,score)
    names=pref_dict.keys()
    return pref.election(names,system_i,pref_dict)

def init_score(pop_size,n_cand):
    population=[]
    for j in range(pop_size):
        if((j%3)==0):
            score_j=[n_cand-i for i in range(n_cand)]
        elif((j%3)==1):
            score_j=np.ones((n_cand,))
            score_j[-1]=0
        else:
            score_j=np.zeros((n_cand,))
            score_j[0]=1
        population.append(score_j)     
    return population

if __name__ == "__main__":
    path="../../VCIP/3DHOI/%s/feats"
    common=[path % "1D_CNN",path % "shapelets"]
    binary=path % "ens/splitII/"
    find_score((common,binary),None)