import numpy as np
import ens,files,learn

class PrefDict(dict):
    def __init__(self, arg=[]):
        super(PrefDict, self).__init__(arg)
    
    def split(self):
        train,test=files.split(self)
        return PrefDict(train),PrefDict(test)

class PrefEnsemble(ens.Ensemble):
    def __init__(self,ensemble=None ,system=None,clf="LR"):
        if(ensemble is None):
            ensemble=ens.Ensemble()
        if(system is None):
            system=majority
        self.ensemble=ensemble
        self.system=system
        self.clf=clf

    def __call__(self,paths):
        result,votes=self.ensemble(paths,binary=False,clf=self.clf)
        pref_dict=to_pref(votes.results)
        test=pref_dict.split()[1]
        names=test.keys()
        y_true=[name_i.get_cat() for name_i in names]
        y_pred=[self.system(pref_dict[name_i] ) for name_i in names]
        return learn.Result(y_true,y_pred,names)

def to_pref(results):
    raw_pref=[ dict(zip(result_i.names,result_i.y_pred))
                for result_i in results]
    names=raw_pref[0].keys()
    pref_dict= PrefDict()
    for name_i in names:
        pref_dict[name_i]=[]
        for raw_j in raw_pref:
            vote_ij=raw_j[name_i]
            pref_ij=np.flip(np.argsort(vote_ij))
            pref_dict[name_i].append(pref_ij)
    return pref_dict

def majority(pref_ij):
    pref_ij=np.array( pref_ij)
    first=pref_ij[:,0]
    unique, counts = np.unique(first, return_counts=True)
    winner= unique[np.argmax(counts)]
    return winner

path="../VCIP/3DHOI/%s/feats"
common=[path % "1D_CNN",path % "shapelets"]
binary=path % "ens/splitI/"
ensemble=PrefEnsemble()
result=ensemble((common,binary))
result.report()