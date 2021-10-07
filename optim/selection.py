import sys
sys.path.append("..")
import diff_evol,gasen,ens,exp

def optim_exp(common,binary):
    algs={}
    algs["diff"]=diff_evol.OptimizeWeights(read=None)#ens.read_multi)
    algs["gasen"]=diff_evol.OptimizeWeights(gasen.Comb,maxiter=10,read=None)
    result={ name_i:select_weights(common,binary,alg_i)
                for name_i,alg_i in algs.items()}
    print(result)

def select_weights(common,binary,optim_alg, threshold=0.05):
    datasets=optim_alg.read(common,binary)
    weights=optim_alg.find_weights(datasets)
    n_clf=weights[weights>threshold].shape[0]
    return n_clf,weights

dir_path="../../3DHOI/"
binary_path="%s/ens/I/feats" % dir_path
base_path="%s/1D_CNN/feats" % dir_path
common=[base_path,"../shape/32_feats"]

input_dict=(base_path,"%s/ens" % dir_path)
optim_exp(common,binary_path)