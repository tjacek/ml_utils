import numpy as np
import ens,files,learn

class PrefDict(dict):
    def __init__(self, arg=[]):
        super(PrefDict, self).__init__(arg)
    
    def n_votes(self):
        return list(self.values())[0].shape[0]

    def n_cand(self):
        return list(self.values())[0].shape[1]

    def split(self):
        train,test=files.split(self)
        return PrefDict(train),PrefDict(test)

    def get_rank(self,name_i,k):
        pref_ij=self[name_i]
        return pref_ij[:,k]       

    def as_counters(self,name_i):
        return [self.count_rank(name_i,k,True) 
                    for k in range(self.n_cand())]

    def count_rank(self,name_i,k,raw_counter=False):
        counter=np.zeros((self.n_cand(),))
        rank_k=self.get_rank(name_i,k)
        for p in rank_k:
            counter[p]+=1
        if(raw_counter):
            return counter
        non_zero=counter[counter!=0].shape[0]
        return np.argsort(counter)[-non_zero:]

class PrefEnsemble(object):#ens.Ensemble):
    def __init__(self,ensemble=None ,system=None):
        ensemble=ens.get_ensemble_helper(ensemble)
        if(system is None):
            system=coombs
        self.ensemble=ensemble
        self.system=system

    def __call__(self,paths):
        result,votes=self.ensemble(paths)
        pref_dict=to_pref(votes.results)
        test=pref_dict.split()[1]
        names=test.keys()
        result=election(names,self.system,pref_dict)
        return result,votes

def election(names,system,pref_dict):
    y_true=[name_i.get_cat() for name_i in names]
    y_pred=[system(name_i,pref_dict ) for name_i in names]
    return learn.Result(y_true,y_pred,names)

def to_pref(results):
    raw_pref=[ dict(zip(result_i.names,result_i.y_pred))
                for result_i in results]
    names=raw_pref[0].keys()
    pref_dict= PrefDict()
    for name_i in names:
        pref_i=[]
        for raw_j in raw_pref:
            vote_ij=raw_j[name_i]
            pref_ij=np.flip(np.argsort(vote_ij))
            pref_i.append(pref_ij)
        pref_dict[name_i]=np.array(pref_i)
    return pref_dict

def majority(name_i,pref_dict):
    first=pref_dict.get_rank(name_i,0)
    unique, counts = np.unique(first, return_counts=True)
    winner= unique[np.argmax(counts)]
    return winner

def borda_count(name_i,pref_dict):
    n_cand=pref_dict.n_cand()
    score=[n_cand-j for j in range(n_cand)]
    return score_rule(name_i,pref_dict, n_cand,score)

def score_rule(name_i,pref_dict, n_cand,score):
    count=np.zeros((n_cand,))
    for j in range(n_cand):
        for vote_k in pref_dict.get_rank(name_i,j):
            count[vote_k]+=score[j]
    return np.argmax(count)

def bucklin(name_i,pref_dict):
    n_cand=pref_dict.n_cand()
    count=np.zeros((n_cand,))
    threshold= np.floor(pref_dict.n_votes()/2)
    for j in range(n_cand):
        for vote_k in pref_dict.get_rank(name_i,j):
            count[vote_k]+=1
        if(np.amax(count)>threshold):
            return np.argmax(count)
    raise Exception("error")

def coombs(name_i,pref_dict):
    counters=pref_dict.as_counters(name_i)
    while(counters):
        threshold=np.floor(np.sum(counters[0])/2)
        if(threshold<np.amax(counters[0])):
            return np.argmax(counters[0])
        if(np.amax(counters[-1])==0):
            counters.pop()
        worst=np.argmax(counters[-1])
        for counter_j in counters:
            counter_j[worst]=0
        if(np.amax(counters[0])==0):
            del counters[0]
        print(len(counters))
    raise Exception(counters)

def major_stats(paths,ensemble=None):
    ensemble=ens.get_ensemble_helper(ensemble)
    result,votes=ensemble(paths)
    pref_dict=to_pref(votes.results)
    threshold=np.ceil(pref_dict.n_votes()/2)
    nonmajor_error=[]
    for name_i in pref_dict.keys():
        pred_i=majority(name_i,pref_dict)
        if(name_i.get_cat()!=pred_i):
            first=pref_dict.get_rank(name_i,0)
            unique, counts = np.unique(first, return_counts=True)
            if(np.amax(counts)<threshold):
                nonmajor_error.append(name_i)
    print(nonmajor_error)

if __name__ == "__main__":
    path="../VCIP/3DHOI/%s/feats"
    common=[path % "1D_CNN","../deep_dtw/dtw"]#path % "shapelets"]
    binary=path % "ens/splitI/"
#    ensemble=PrefEnsemble()
#    result=ensemble((common,binary))
#    result.report()
#    print(result.get_cf())
    major_stats((common,binary))