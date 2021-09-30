import sys
sys.path.append("..")
import diff_evol,gasen,ens,exp
 
def optim_exp(common_path,binary_path,out_path=None):
    gasen_ensemble=diff_evol.OptimizeWeights(
    	                gasen.Comb,
                        maxiter=10,read=ens.read_multi)
    diff_optim=diff_evol.OptimizeWeights()
    simple_ensemble=ens.Ensemble(ens.read_multi)
    all_ens={"soft":simple_ensemble,"diff":diff_optim}
#                "gasen":gasen_ensemble}
    ens_exp=exp.MultiEnsembleExp(all_ens)
    input_dict={"common":common_path,"binary":binary_path}
    ens_exp(input_dict,out_path)

dir_path="../../3DHOI/"
binary_path="%s/ens/I/feats" % dir_path
base_path="%s/1D_CNN/feats" % dir_path
dtw_path="../../deep_dtw/dtw"
ae_path="../../best2/3_layers/feats"
common=[base_path,ae_path]
optim_exp(common,binary_path,out_path=None)