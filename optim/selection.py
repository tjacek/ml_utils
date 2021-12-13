import sys
sys.path.append("..")
import diff_evol,gasen,ens,pref,optim

def select_clfs(common,binary,read_type=None):
    optim_alg=diff_evol.OptimizeWeights(gasen.Corl,maxiter=10,read=read_type)
    n_clf,s_clf,weights=select_weights(common,binary,optim_alg, threshold=0.05)
    result=optim_alg.eval_weights(weights,(common,binary))
    print(n_clf)
    print(s_clf)
    print(weights)
    acc=result.get_acc()
    return s_clf,acc

def select_weights(common,binary,optim_alg, threshold=0.05):
    datasets=optim_alg.read(common,binary)
    weights=optim_alg.find_weights(datasets)
    s_clf=[i  for i,weights_i in enumerate(weights)
            if(weights_i>threshold)]
    n_clf=len(s_clf)
    return n_clf,s_clf,weights

if __name__ == "__main__":
    path="../../VCIP/3DHOI/%s/feats"
    common=[path % "1D_CNN","../../deep_dtw/dtw"]#path % "shapelets"]
    binary=path % "ens/splitII/"
    s_clf=select_clfs(common,binary,read_type=None)
    print(s_clf)   
    optim.optim_exp(common,binary,out_path=None,
                 read_type=None,s_clf=s_clf)