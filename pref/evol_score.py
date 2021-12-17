import sys
sys.path.append("..")
sys.path.append("../optim")
import numpy as np
from optim.selection import select_clfs
import pref,ens,exp
import optim_algs,systems,get_data

class PrefExp(object):
    def __init__(self,alg_optim,get_data=None):
        if(get_data is None):
            get_data=get_pref_dict
        self.alg_optim=alg_optim
        self.get_data=get_data

    def __call__(self,paths,n_iters):
        old_results,new_results=[],[]
        for i in range(n_iters):
            print("Epoch %d" % i)
            old_i,s_clf=select_clfs(paths,read_type=None)
            new_i,score_i=self.find_score(paths,s_clf)
            old_results.append(old_i)
            new_results.append(new_i)
        acc=[result_i.get_acc() for result_i in new_results]
        k=np.argmax(acc)
        return old_results[k],new_results[k]

    def find_score(self,paths,ensemble):
        ensemble=ens.get_ensemble_helper(ensemble)
        pref_dict=self.get_data(paths,ensemble) 
        n_cand=pref_dict.n_cand()
        def loss_fun(score):
            result=eval_score(score,n_cand,pref_dict)
            acc=result.get_acc()
            print(acc)
            return (1.0-acc)
        score=self.alg_optim(loss_fun,n_cand)
        print(score)
        result=eval_score(score,n_cand,pref_dict)
        return result,score

def paths_exp(all_paths,out_path,pref_exp,n_iters=5):
    lines=[]
    def helper(desc_i,result_i):
        line_i="%s,%s" % (desc_i,exp.get_metrics(result_i))
        lines.append(line_i)
    for paths_i in all_paths:
        old,new=pref_exp(paths_i,n_iters)
        desc_i=exp.paths_desc(paths_i)
        helper("old,%s"%desc_i,old)
        helper("new,%s"%desc_i,new)
        print(lines)
    exp.save_lines(lines,out_path)

def eval_score(score,n_cand,pref_dict):
    def system_i(name_i,pref_dict):
        return systems.score_rule(name_i,pref_dict, n_cand,score)
    names=pref_dict.keys()
    return pref.election(names,system_i,pref_dict)

def get_paths():
    path="../../VCIP/3DHOI/%s/feats"
    common=[path % "1D_CNN","../../deep_dtw/dtw"]
#                path % "shapelets"]
    return [(common,path % "ens/splitI"),(common,path % "ens/splitII")]    

if __name__ == "__main__":    
    optim=optim_algs.GenAlg()#init_type="borda")
    paths=get_paths()
#    print(exp.paths_desc(paths[0]))
    pref_exp=PrefExp(optim,get_data.validate)
    paths_exp(paths,"test2",pref_exp,2)