import sys
sys.path.append("..")
import diff_evol,gasen,ens,pref,optim

def select_clfs(common_path,binary_path,read_type=None):
    optim_alg=diff_evol.OptimizeWeights(gasen.Corl,maxiter=10,read=read_type)
    n_clf,clfs,weights=select_weights(common,binary,optim_alg, threshold=0.05)
    print(n_clf)
    print(clfs)
    print(weights)
    return clfs

def select_weights(common,binary,optim_alg, threshold=0.05):
    datasets=optim_alg.read(common,binary)
    weights=optim_alg.find_weights(datasets)
    clfs=[i  for i,weights_i in enumerate(weights)
            if(weights_i>threshold)]
    n_clf=len(clfs)
    return n_clf,clfs,weights


if __name__ == "__main__":
    path="../../VCIP/3DHOI/%s/feats"
    common=[path % "1D_CNN",path % "shapelets"]
    binary=path % "ens/splitI/"
    s_clf=select_clfs(common,binary,read_type=None)
    optim.optim_exp(common,binary,out_path=None,
                 read_type=None,s_clf=s_clf)