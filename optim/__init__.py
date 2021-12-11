import sys
sys.path.append("..")
import diff_evol,gasen,ens,exp,pref
 
def optim_exp(common_path,binary_path,out_path=None,read_type=None):
    all_ens={}
#    all_voting_methods(all_ens ,read_type=read_type ,desc="")
    all_system(all_ens ,read_type=read_type ,desc="")
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

def all_system(all_ens ,read_type=None ,desc=""):
    ensemble=ens.Ensemble(read_type)
    algs={"borda_count":pref.PrefEnsemble(ensemble,pref.borda_count),
           "coombs":pref.PrefEnsemble(ensemble,pref.coombs),
           "bucklin":pref.PrefEnsemble(ensemble,pref.bucklin)}
    for name_i,alg_i in algs.items():
        all_ens["%s%s" % (name_i,desc)]=alg_i
    return all_ens

if __name__ == "__main__":
    path="../../VCIP/3DHOI/%s/feats"
    common=[path % "1D_CNN",path % "shapelets"]
    binary=path % "ens/splitII/"
    optim_exp(common,binary,out_path="3DHOI_II",read_type=None)