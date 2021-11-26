import sys
sys.path.append("..")
import diff_evol,gasen,ens,exp
 
def optim_exp(common_path,binary_path,out_path=None,read_type=None):
    all_ens={}
    all_voting_methods(all_ens ,read_type=read_type ,desc="")
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

common=[None,"../../cc2/segm2/feats"]
binary="../../cc2/ens/feats"
optim_exp(common,binary,out_path=None,read_type=ens.read_multi)