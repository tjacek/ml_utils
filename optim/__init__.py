import sys
sys.path.append("..")
import diff_evol,gasen,ens,exp
 
def optim_exp(common_path,binary_path,out_path=None):
    all_ens={}
    all_voting_methods(all_ens ,read_type=None ,desc="")
    ens_exp=exp.MultiEnsembleExp(all_ens)
    input_dict=(common_path,binary_path)
    ens_exp(input_dict,out_path)

def all_voting_methods(all_ens ,read_type=None ,desc=""):
    ensemble=ens.Ensemble(read_type)
    algs={"soft":ensemble,"hard":ens.EnsembleHelper(ensemble,binary=False),
           "diff":diff_evol.OptimizeWeights(read=read_type),
           "gasen":diff_evol.OptimizeWeights(gasen.Corl,maxiter=10,read=read_type) }
    for name_i,alg_i in algs.items():
        all_ens["%s%s" % (name_i,desc)]=alg_i
    return all_ens

#def single_exp(input_dict):
#    exp=make_exp(None)
#    exp(input_dict,"unified.csv")
#    exp=make_exp(ens.read_multi)
#    exp(input_dict,"distinct.csv")

#def gen(input_dict):
#    paths=["../../deep_dtw/dtw",
#            "../../best2/3_layers/feats"]
#    common,binary=input_dict
#    for path_j in paths:
#        for split_i in ["I","II"]:
#            common_j=[common,path_j]
#            binary_i="%s/%s/feats" % (binary,split_i)
#            desc_i="%s,%s" % (split_i,path_j.split("/")[-2])
#            yield desc_i,(common_j,binary_i)

common="../../common/feats"
binary="../../cc2/ens/feats"
optim_exp(common,binary,out_path="florence2.txt")