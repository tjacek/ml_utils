import sys
sys.path.append("..")
sys.path.append("../optim")
import numpy as np
from optim.selection import select_clfs
import pref,ens,exp
import optim_algs,systems,get_data,learn,files
import loss

class PrefExp(object):
    def __init__(self,alg_optim,read=None):
        if(read is None):
            read=get_data.person_dict
        if(alg_optim=="genetic"):
            alg_optim=optim_algs.GenAlg()
        if(alg_optim=="swarm"):
            alg_optim=optim_algs.SwarmAlg()
        self.alg_optim=alg_optim
        self.read=read
        self.loss=loss.AucLoss

    def __call__(self,paths,n_iters,ensemble=None):
        ensemble=ens.get_ensemble_helper(ensemble)
        old_result=ensemble(paths)[0]
        old_acc=old_result.get_acc()
        diff_acc,new_results,calls=[],[],[]
        for i in range(n_iters):
            print("Epoch %d" % i)
            new_i,score_i,n_calls_i=self.find_score(paths,ensemble)
            new_results.append(new_i)
            calls.append(n_calls_i)
            diff_acc.append(new_i.get_acc()-old_acc)
        best=np.argmax(diff_acc)
        return new_results[best],calls[best]

    def find_score(self,paths,ensemble):
        ensemble=ens.get_ensemble_helper(ensemble)
        train_dict,test_dict=self.read(paths,ensemble)
        loss_fun=self.loss(train_dict) 
        n_cand=train_dict.n_cand()
        score=self.alg_optim(loss_fun,n_cand)
        print(score)
        result=pref.eval_score(score,test_dict)
        return result,score,loss_fun.n_calls

    def __str__(self):
        alg=self.alg_optim.__class__.__name__
        init=self.alg_optim.init_type
        return "%s,%s" % (alg,init)


def all_paths_exp(all_paths,all_clf,out_path,n_iters=5):
    lines=[]
    for alg_j in all_algs():
        pref_j=PrefExp(alg_j,get_data.person_dict)
        for paths_i,clf_i in zip(all_paths,all_clf):
            desc_i="%s,%s,%s" % (str(pref_j),exp.paths_desc(paths_i),str(clf_i))
            alg_j.n_calls=0
            result_i,calls_i=pref_j(paths_i,n_iters,clf_i)
            line_i="Yes,%s,%d,%s" % (desc_i,calls_i,exp.get_metrics(result_i))
            lines.append(line_i)
            print(lines)
    lines=exp.order_lines(lines)
    exp.save_lines(lines,out_path)

def all_algs(maxiter=1,pop_size=100):
    alg_desc=[(optim_algs.GenAlg,['latinhypercube','borda',"borda_mixed"]),
          (optim_algs.SwarmAlg,['random','borda',"borda_mixed"])]
    for alg_i,inits_i in alg_desc:
        for init_j in inits_i:
            yield alg_i(maxiter=maxiter,pop_size=pop_size,init_type=init_j)

def get_paths():
    path="../../VCIP/3DHOI/%s/feats"
    common=[path % "1D_CNN","../../deep_dtw/dtw"]
#                path % "shapelets"]
    for binary_i in ["ens/splitI","ens/splitII"]:
        yield (common,path % binary_i)
#    return [(common,path % "ens/splitI"),(common,path % "ens/splitII")]    

def check_calls(paths):
    lines=[]
    for pref_i in [PrefExp ("genetic"),PrefExp('swarm')]:
        result,score,n_calls,=pref_i.find_score(paths,None)
        lines.append("%.4f,%d" % (result.get_acc(),n_calls))
    print(lines)

def score_exp(out_dict,paths,clfs,n_iters=10):
    files.make_dir(out_dict)
    all_paths_exp(paths,clfs,f'{out_dict}/selected',n_iters)
    all_paths_exp(paths,[None,None],f"{out_dict}/full",n_iters)

if __name__ == "__main__":    
    paths=get_paths()
    all_clf=[[0, 1, 2, 8, 9, 10, 11],[3,9,11]]
    score_exp("test",paths,all_clf,n_iters=1)