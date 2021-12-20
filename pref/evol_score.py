import sys
sys.path.append("..")
sys.path.append("../optim")
import numpy as np
from optim.selection import select_clfs
import pref,ens,exp
import optim_algs,systems,get_data

class PrefExp(object):
    def __init__(self,alg_optim,get_data=None,selected=False):
        if(get_data is None):
            get_data=get_data.person_dict
        self.alg_optim=alg_optim
        self.get_data=get_data
        self.selected=selected

    def __call__(self,paths,n_iters):
        if(self.selected):
            return self.selected_exp(paths,n_iters)
        else:
            return self.full_exp(paths,n_iters)

    def selected_exp(self,paths,n_iters):
        old_results,new_results,s_clfs=[],[],[]
        for i in range(n_iters):
            print("Epoch %d" % i)
            old_i,s_clf_i=select_clfs(paths,read_type=None)
            new_i,score_i=self.find_score(paths,s_clf_i)
            old_results.append(old_i)
            new_results.append(new_i)
            s_clfs.append( s_clf_i)
        diff_acc,k=get_diff_acc(new_results,old_results)
        print("diff acc %.5f" % diff_acc)
        return old_results[k],new_results[k],s_clfs[k]

    def full_exp(self,paths,n_iters):
        old_results,new_results=[],[]
        for i in range(n_iters):
            print("Epoch %d" % i)
            old_i=ens.get_ensemble_helper(None)(paths)[0]
            new_i,score_i=self.find_score(paths,None)
            old_results.append(old_i)
            new_results.append(new_i)
        diff_acc,k=get_diff_acc(new_results,old_results)
        print("diff acc %.5f" % diff_acc)
        return old_results[k],new_results[k],len(score_i)

    def find_score(self,paths,ensemble):
        ensemble=ens.get_ensemble_helper(ensemble)
        train_dict,test_dict=self.get_data(paths,ensemble) 
        n_cand=train_dict.n_cand()
        def loss_fun(score):
            result=eval_score(score,n_cand,train_dict)
            acc=result.get_acc()
            print(acc)
            return (1.0-acc)
        score=self.alg_optim(loss_fun,n_cand)
        print(score)
        result=eval_score(score,n_cand,test_dict)
        return result,score

    def __str__(self):
        alg=self.alg_optim.__class__.__name__
        init=self.alg_optim.init_type
        return "%s,%s" % (alg,init)

def get_diff_acc(new_results,old_results):
    diff_acc=[ (new_i.get_acc()-old_i.get_acc()) 
                    for new_i,old_i in zip(new_results,old_results)]
    k=np.argmax(diff_acc)
    return diff_acc[k],k

#def paths_exp(all_paths,out_path,all_exps,n_iters=5):
#    if(type(all_exps)!=list):
#        all_exps=[all_exps]
#    lines=[]
#    def helper(desc_i,result_i):
#        line_i="%s,%s" % (desc_i,exp.get_metrics(result_i))
#        lines.append(line_i)
#    for pref_exp_j in all_exps:
#        for paths_i in all_paths:
#            old,new,s_clf=pref_exp_j(paths_i,n_iters)
#            desc_i="%s,%s,%s" % (str(pref_exp_j),exp.paths_desc(paths_i),str(s_clf))
#            helper("old,%s"%desc_i,old)
#            helper("new,%s"%desc_i,new)
#            print(lines)
#    exp.save_lines(lines,out_path)

def eval_score(score,n_cand,pref_dict):
    def system_i(name_i,pref_dict):
        return systems.score_rule(name_i,pref_dict,n_cand,score)
    names=pref_dict.keys()
    return pref.election(names,system_i,pref_dict)

def all_algs_exp(paths,s_clf,out_path):
    lines=[]
    for alg_i in all_algs():
        pref_i=PrefExp(alg_i,get_data.person_dict,False)
        result_i,score_i=pref_i.find_score(paths,s_clf)
        desc_i="%s,%s,%s" % (str(pref_i),exp.paths_desc(paths),str(score_i))
        line_i="%s,%s" % (desc_i,exp.get_metrics(result_i))
        lines.append(line_i)
        print(lines)
    exp.save_lines(lines,out_path)

def all_algs():
    algs=[optim_algs.GenAlg(init_type=init_i)  
             for init_i in ['latinhypercube','borda',"borda_mixed"]]
    algs+=[optim_algs.SwarmAlg(init_type=init_i)  
             for init_i in ['random','borda',"borda_mixed"]]    
    return algs

def get_paths():
    path="../../VCIP/3DHOI/%s/feats"
    common=[path % "1D_CNN","../../deep_dtw/dtw"]
#                path % "shapelets"]
    return [(common,path % "ens/splitII")]#,(common,path % "ens/splitII")]    

if __name__ == "__main__":    
    optim=optim_algs.GenAlg()#init_type="borda")
    paths=get_paths()
#    print(exp.paths_desc(paths[0]))
    all_algs_exp(paths[0],[3, 9, 11],"bestII")
    