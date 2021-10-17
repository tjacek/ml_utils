import sys
sys.path.append("..")
import diff_evol,gasen,ens,exp
 
def optim_exp(common_path,binary_path,out_path=None):
    gasen_ensemble=diff_evol.OptimizeWeights(
    	                gasen.Comb,
                        maxiter=10,read=ens.read_multi)
    diff_optim=diff_evol.OptimizeWeights(read=ens.read_multi)
    soft_ensemble=ens.Ensemble(ens.read_multi)
    hard_ensemble=ens.EnsembleHelper(soft_ensemble,binary=False)
    all_ens={"soft,distinct":soft_ensemble,"hard,distinct":hard_ensemble,
        "diff,distinct":diff_optim,"gasen,distinct":gasen_ensemble}
#    all_ens={}
    ensemble=ens.Ensemble(None)
    all_ens["soft,unified"]=ensemble
    all_ens["hard,unified"]=ens.EnsembleHelper(ensemble,binary=False)
    all_ens["diff,unified"]=diff_evol.OptimizeWeights(read=None)
    all_ens["gasen,unified"]=diff_evol.OptimizeWeights(gasen.Comb,maxiter=10,read=None)
    ens_exp=exp.MultiEnsembleExp(all_ens)
    input_dict=(common_path,binary_path)
    ens_exp(input_dict,out_path)

def single_exp(input_dict):
    exp=make_exp(None)
    exp(input_dict,"unified.csv")
    exp=make_exp(ens.read_multi)
    exp(input_dict,"distinct.csv")

def make_exp(read=None):
    raw= ens.Ensemble(read)
    helper=ens.EnsembleHelper(raw,binary=True)
    return exp.EnsembleExp(helper,gen)

def gen(input_dict):
    paths=["../../deep_dtw/dtw",
            "../../best2/3_layers/feats"]
    common,binary=input_dict
    for path_j in paths:
        for split_i in ["I","II"]:
            common_j=[common,path_j]
            binary_i="%s/%s/feats" % (binary,split_i)
            desc_i="%s,%s" % (split_i,path_j.split("/")[-2])
            yield desc_i,(common_j,binary_i)

dir_path="../../3DHOI/"
binary_path="%s/ens/II/feats" % dir_path
base_path="%s/1D_CNN/feats" % dir_path
dtw_path="../../deep_dtw/dtw"
ae_path="../../best2/3_layers/feats"
common=[base_path,"../shape/32_feats"]

input_dict=(base_path,"%s/ens" % dir_path)
#single_exp(input_dict)
optim_exp(common,binary_path,out_path="I_shaplets.txt")