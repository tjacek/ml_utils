import sys
sys.path.append("..")
sys.path.append("../optim")
import numpy as np
from optim.selection import select_clfs
import pref,ens,exp
import optim_algs,systems

def paths_exp(all_paths,out_path,optim,n_iters=5):
    lines=[]
    def helper(desc_i,result_i):
        line_i="%s,%s" % (desc_i,exp.get_metrics(result_i))
        lines.append(line_i)
    for paths_i in all_paths:
        old,new=score_exp(paths_i,n_iters,optim)
        desc_i=paths_i[1].split("/")[-1]
        helper("old,%s"%desc_i,old)
        helper("new,%s"%desc_i,new)
    exp.save_lines(lines,out_path)

def score_exp(paths,n_iters,optim):
    old_results,new_results=[],[]
    for i in range(n_iters):
        print("Epoch %d" % i)
        old_i,s_clf=select_clfs(paths,read_type=None)
        new_i,score_i=find_score(paths,s_clf,optim)
        old_results.append(old_i)
        new_results.append(new_i)
    acc=[result_i.get_acc() for result_i in new_results]
    k=np.argmax(acc)
    return old_results[k],new_results[k]
#    new_results[k].report()
#    print(old_results[k].get_acc())

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
        return systems.score_rule(name_i,pref_dict, n_cand,score)
    names=pref_dict.keys()
    return pref.election(names,system_i,pref_dict)

def get_paths():
    path="../../VCIP/3DHOI/%s/feats"
    common=[path % "1D_CNN","../../deep_dtw/dtw"]
#                path % "shapelets"]
#    binary=path % "ens/splitI/"
    return [(common,path % "ens/splitI/"),(common,path % "ens/splitII/")]    

if __name__ == "__main__":    
    optim=optim_algs.GenAlg()#init_type="borda")
    paths=get_paths()
    paths_exp(paths,"test",optim,5)