import sys
sys.path.append("..")
import ens,pref

class ScoreEnsemble(object):
    def __init__(self,ensemble=None):
        ensemble=ens.get_ensemble_helper(ensemble)
        self.ensemble=ensemble

    def __call__(self,paths):    
        result,votes=self.ensemble(paths)
        pref_dict=pref.to_pref(votes.results)
        pref_dict=pref_dict.split()[1]
        print(len(pref_dict))


if __name__ == "__main__":
    path="../../VCIP/3DHOI/%s/feats"
    common=[path % "1D_CNN",path % "shapelets"]
    binary=path % "ens/splitII/"
    ensemble=ScoreEnsemble()
    ensemble((common,binary))