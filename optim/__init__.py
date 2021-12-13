import sys
sys.path.append("..")
import diff_evol,gasen,ens,exp,pref
 
def optim_exp(common_path,binary_path,out_path=None,
    read_type=None,s_clf=None):
    all_ens={}
    all_voting_methods(all_ens ,read_type=read_type ,desc="")
    all_system(all_ens ,read_type=read_type ,desc="",s_clf=s_clf)
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

def all_system(all_ens ,read_type=None ,desc="",s_clf=None):
    ensemble=ens.Ensemble(read_type)
    def helper(system_i):
        return pref.PrefEnsemble(s_clf,system_i)
    algs={"borda_count":helper(pref.borda_count),
           "coombs":helper(pref.coombs),
           "bucklin":helper(pref.bucklin)}
    for name_i,alg_i in algs.items():
        all_ens["%s%s" % (name_i,desc)]=alg_i
    return all_ens

if __name__ == "__main__":
    path="../../VCIP/3DHOI/%s/feats"
    common=[path % "1D_CNN","../../deep_dtw/dtw"]#path % "shapelets"]
    binary=path % "ens/splitII/"
    optim_exp(common,binary,out_path="dtw_I",
        read_type=None)#s_clf=[2, 9, 11])